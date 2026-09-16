package ui

import (
	"testing"

	"github.com/kidus-der/acceptrate/tui/internal/api"
)

func stats(k int, alpha, tokS float64, busy bool, windows int) api.Stats {
	return api.Stats{Model: "m", Draft: "d", Busy: busy, KCurrent: k,
		AlphaEWMA: &alpha, TokSRecent: &tokS, WindowsTotal: windows}
}

func TestApplyStatsFirstSnapshotSnapsKWithoutAnimating(t *testing.T) {
	s := ApplyStats(State{}, stats(4, 0.5, 20, false, 0))

	if !s.Connected || s.KTarget != 4 || s.KShown != 4 || s.KPrev != 0 {
		t.Errorf("state = %+v", s)
	}
	if s.Err != "" {
		t.Errorf("connecting clears the error, got %q", s.Err)
	}
}

func TestApplyStatsKChangeRecordsPreviousAndKeepsShownForTheSpring(t *testing.T) {
	s := ApplyStats(State{}, stats(4, 0.5, 20, true, 10))

	s = ApplyStats(s, stats(6, 0.7, 28, true, 11))

	if s.KTarget != 6 || s.KPrev != 4 || s.KShown != 4 {
		t.Errorf("state = %+v", s)
	}
	s = ApplyStats(s, stats(6, 0.72, 29, true, 12))
	if s.KPrev != 4 || s.KTarget != 6 {
		t.Errorf("unchanged K keeps the adaptation note: %+v", s)
	}
}

func TestApplyStatsGrowsTheThroughputScale(t *testing.T) {
	s := State{Baseline: 16.6}

	s = ApplyStats(s, stats(4, 0.5, 20, true, 1))
	if s.TokSScale < 20 || s.TokSScale < 16.6 {
		t.Errorf("scale %v should cover both tok/s and the baseline", s.TokSScale)
	}
	s = ApplyStats(s, stats(4, 0.5, 31, true, 2))
	if s.TokSScale < 31 {
		t.Errorf("scale %v should grow with tok/s", s.TokSScale)
	}
	s = ApplyStats(s, stats(4, 0.5, 12, true, 3))
	if s.TokSScale < 31 {
		t.Errorf("scale %v should not shrink (the bar would jump)", s.TokSScale)
	}
}

func TestApplyChunkAccumulatesThenCommitsOnDone(t *testing.T) {
	s, _ := Submit(State{Connected: true}, "hi")

	s = ApplyChunk(s, api.ChatEvent{Text: "Hel"})
	s = ApplyChunk(s, api.ChatEvent{Text: "lo"})
	if s.Pending != "Hello" || !s.Streaming {
		t.Fatalf("mid-stream state = %+v", s)
	}
	s = ApplyChunk(s, api.ChatEvent{Finish: "stop"})
	s = ApplyChunk(s, api.ChatEvent{Done: true})

	if s.Streaming || s.Pending != "" {
		t.Errorf("done should end streaming: %+v", s)
	}
	want := []api.Message{{Role: "user", Content: "hi"}, {Role: "assistant", Content: "Hello"}}
	if len(s.Transcript) != 2 || s.Transcript[0] != want[0] || s.Transcript[1] != want[1] {
		t.Errorf("transcript = %+v", s.Transcript)
	}
}

func TestApplyChunkErrorEndsTheTurnAndKeepsPartialText(t *testing.T) {
	s, _ := Submit(State{Connected: true}, "hi")
	s = ApplyChunk(s, api.ChatEvent{Text: "partial"})

	s = ApplyChunk(s, api.ChatEvent{Err: "generation failed; see server log"})

	if s.Streaming || s.Err == "" {
		t.Errorf("error should stop streaming and surface: %+v", s)
	}
	if len(s.Transcript) != 2 || s.Transcript[1].Content != "partial" {
		t.Errorf("partial text should be committed: %+v", s.Transcript)
	}
}

func TestSubmitRules(t *testing.T) {
	tests := []struct {
		name  string
		state State
		text  string
		ok    bool
	}{
		{"plain text", State{Connected: true}, "hello", true},
		{"blank is ignored", State{Connected: true}, "   ", false},
		{"while streaming is ignored", State{Connected: true, Streaming: true}, "x", false},
		{"disconnected still queues (the server may come back)", State{}, "x", true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := Submit(tt.state, tt.text)
			if ok != tt.ok {
				t.Fatalf("ok = %v, want %v", ok, tt.ok)
			}
			if ok && (!got.Streaming || got.Transcript[len(got.Transcript)-1].Content != "hello" && tt.text == "hello") {
				t.Errorf("submitted state = %+v", got)
			}
			if !ok && len(got.Transcript) != len(tt.state.Transcript) {
				t.Errorf("rejected submit must not change the transcript")
			}
		})
	}
}

func TestSubmitDoesNotMutateTheCallerTranscript(t *testing.T) {
	original := State{Transcript: []api.Message{{Role: "user", Content: "a"}}}

	got, _ := Submit(original, "b")

	if len(original.Transcript) != 1 || len(got.Transcript) != 2 {
		t.Errorf("original %d, new %d", len(original.Transcript), len(got.Transcript))
	}
}

func TestStepSpringAnimatesThenReportsSettled(t *testing.T) {
	a := NewAnimator(30, false)
	s := State{KShown: 4, KTarget: 6}

	s, animating := StepSpring(s, a)
	if !animating || s.KShown <= 4 {
		t.Fatalf("first step: animating=%v shown=%v", animating, s.KShown)
	}
	for i := 0; i < 90 && animating; i++ {
		s, animating = StepSpring(s, a)
	}
	if animating || s.KShown != 6 {
		t.Errorf("after settle: animating=%v shown=%v (want exactly 6)", animating, s.KShown)
	}
}

func TestStatusLine(t *testing.T) {
	tests := []struct {
		name  string
		state State
		want  string
	}{
		{"disconnected", State{}, "draft window #0 · connecting…"},
		{"idle", State{Connected: true, Stats: stats(4, 0.5, 20, false, 12)}, "draft window #12 · idle"},
		{"generating", State{Connected: true, Streaming: true, Stats: stats(4, 0.5, 20, true, 12)}, "draft window #12 · generating"},
		{"server busy elsewhere", State{Connected: true, Stats: stats(4, 0.5, 20, true, 12)}, "draft window #12 · busy (another client)"},
		{"error", State{Connected: true, Err: "boom", Stats: stats(4, 0.5, 20, false, 3)}, "draft window #3 · error: boom"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := StatusLine(tt.state); got != tt.want {
				t.Errorf("got %q, want %q", got, tt.want)
			}
		})
	}
}
