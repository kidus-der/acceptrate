package ui

import (
	"strings"
	"testing"
	"time"

	tea "charm.land/bubbletea/v2"
	"github.com/charmbracelet/x/ansi"

	"github.com/kidus-der/acceptrate/tui/internal/api"
	"github.com/kidus-der/acceptrate/tui/internal/mock"
	"github.com/kidus-der/acceptrate/tui/theme"
)

func newTestModel(t *testing.T) Model {
	t.Helper()
	m := New(Config{Styles: NewStyles(theme.Dark()), FPS: 30, Baseline: 16.6})
	m = update(t, m, tea.WindowSizeMsg{Width: 80, Height: 24})
	return m
}

func update(t *testing.T, m Model, msg tea.Msg) Model {
	t.Helper()
	next, _ := m.Update(msg)
	got, ok := next.(Model)
	if !ok {
		t.Fatalf("Update returned %T", next)
	}
	return got
}

func plainView(m Model) string {
	return ansi.Strip(m.View().Content)
}

func TestViewShowsTitleStatusAndGaugesFromStats(t *testing.T) {
	m := newTestModel(t)
	step := mock.Demo().Steps[5]

	m = update(t, m, StatsMsg{Name: "stats", Stats: step.Stats})
	view := plainView(m)

	for _, want := range []string{
		"acceptrate chat — " + step.Stats.Model + " ← draft " + step.Stats.Draft,
		"draft window #6 · busy (another client)",
		"acceptance", "throughput", "tok/s", "draft depth", "K = 4",
		"windows 6",
	} {
		if !strings.Contains(view, want) {
			t.Errorf("view lacks %q:\n%s", want, view)
		}
	}
	for i, line := range strings.Split(view, "\n") {
		if w := ansi.StringWidth(line); w > 80 {
			t.Errorf("line %d is %d wide: %q", i, w, line)
		}
	}
	if h := strings.Count(view, "\n") + 1; h != 24 {
		t.Errorf("view is %d lines, want 24 (full height)", h)
	}
}

func TestEnterSubmitsAndClearsInputChunksStreamIntoTranscript(t *testing.T) {
	m := newTestModel(t)
	m = update(t, m, StatsMsg{Name: "stats", Stats: mock.Demo().Idle})
	m.input.SetValue("write a haiku")

	m = update(t, m, tea.KeyPressMsg{Code: tea.KeyEnter})

	if !m.state.Streaming || m.input.Value() != "" {
		t.Fatalf("after enter: streaming=%v input=%q", m.state.Streaming, m.input.Value())
	}
	m = update(t, m, ChunkMsg{Event: api.ChatEvent{Text: "Old pond"}})
	if view := plainView(m); !strings.Contains(view, "write a haiku") || !strings.Contains(view, "Old pond") {
		t.Errorf("transcript missing turns:\n%s", view)
	}
	if !strings.Contains(plainView(m), "generating") {
		t.Errorf("status should say generating while streaming")
	}
	m = update(t, m, ChunkMsg{Event: api.ChatEvent{Done: true}})
	if m.state.Streaming || len(m.state.Transcript) != 2 {
		t.Errorf("after done: %+v", m.state)
	}
}

func TestEnterOnBlankInputIsIgnored(t *testing.T) {
	m := newTestModel(t)

	m = update(t, m, tea.KeyPressMsg{Code: tea.KeyEnter})

	if m.state.Streaming || len(m.state.Transcript) != 0 {
		t.Errorf("blank enter changed state: %+v", m.state)
	}
}

func TestCtrlCQuits(t *testing.T) {
	m := newTestModel(t)

	_, cmd := m.Update(tea.KeyPressMsg{Code: 'c', Mod: tea.ModCtrl})

	if cmd == nil {
		t.Fatal("no command")
	}
	if _, ok := cmd().(tea.QuitMsg); !ok {
		t.Errorf("ctrl+c should quit, got %T", cmd())
	}
}

func TestKChangeStartsTheSpringAndTicksUntilSettled(t *testing.T) {
	m := newTestModel(t)
	m = update(t, m, StatsMsg{Name: "stats", Stats: stats(4, 0.5, 20, true, 1)})

	next, cmd := m.Update(StatsMsg{Name: "stats", Stats: stats(6, 0.7, 28, true, 2)})
	m = next.(Model)

	if !m.animating || cmd == nil {
		t.Fatalf("K change should start animating with a tick: animating=%v cmd=%v", m.animating, cmd)
	}
	if !strings.Contains(plainView(m), "↑ adapting from 4") {
		t.Errorf("view should show the adaptation note:\n%s", plainView(m))
	}
	ticks := 0
	for m.animating && ticks < 200 {
		m = update(t, m, TickMsg(time.Now()))
		ticks++
	}
	if m.animating || m.state.KShown != 6 {
		t.Errorf("after %d ticks: animating=%v shown=%v", ticks, m.animating, m.state.KShown)
	}
	if ticks < 3 || ticks > 30 {
		t.Errorf("settled in %d ticks, want a visible sub-second spring", ticks)
	}
}

func TestReducedMotionSnapsK(t *testing.T) {
	m := New(Config{Styles: NewStyles(theme.Dark()), FPS: 30, ReducedMotion: true})
	m = update(t, m, tea.WindowSizeMsg{Width: 80, Height: 24})
	m = update(t, m, StatsMsg{Name: "stats", Stats: stats(4, 0.5, 20, true, 1)})

	m = update(t, m, StatsMsg{Name: "stats", Stats: stats(6, 0.7, 28, true, 2)})
	m = update(t, m, TickMsg(time.Now()))

	if m.animating || m.state.KShown != 6 {
		t.Errorf("reduced motion: animating=%v shown=%v", m.animating, m.state.KShown)
	}
}

func TestStreamErrorsSurfaceInStatus(t *testing.T) {
	m := newTestModel(t)
	m = update(t, m, StatsMsg{Name: "stats", Stats: mock.Demo().Idle})

	m = update(t, m, StatsErrMsg{Err: errString("connection refused")})
	if m.state.Connected || !strings.Contains(plainView(m), "connection refused") {
		t.Errorf("stats error should disconnect and show:\n%s", plainView(m))
	}

	m = update(t, m, StatsMsg{Name: "heartbeat", Stats: mock.Demo().Idle})
	if !m.state.Connected || strings.Contains(plainView(m), "connection refused") {
		t.Errorf("a heartbeat reconnects and clears the error")
	}
}

func TestStatsOnlyHidesTheChatAndIgnoresEnter(t *testing.T) {
	m := New(Config{Styles: NewStyles(theme.Dark()), FPS: 30, StatsOnly: true})
	m = update(t, m, tea.WindowSizeMsg{Width: 80, Height: 24})
	m.input.SetValue("hello")

	m = update(t, m, tea.KeyPressMsg{Code: tea.KeyEnter})

	if m.state.Streaming || strings.Contains(plainView(m), "> ") {
		t.Errorf("stats-only must not submit or show the prompt:\n%s", plainView(m))
	}
}

type errString string

func (e errString) Error() string { return string(e) }
