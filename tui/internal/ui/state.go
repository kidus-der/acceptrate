package ui

import (
	"fmt"
	"strings"

	"github.com/kidus-der/acceptrate/tui/internal/api"
)

// minTokSScale keeps the throughput bar readable before the first window.
const minTokSScale = 10.0

// Status labels for the second header line.
const (
	statusConnecting = "connecting…"
	statusIdle       = "idle"
	statusGenerating = "generating"
	statusBusyOther  = "busy (another client)"
)

// State is everything the panel shows, kept as a value so every transition
// returns a new copy and the tests need no terminal.
type State struct {
	Stats     api.Stats
	Connected bool
	Err       string

	Transcript []api.Message
	Pending    string
	Streaming  bool

	// Draft-depth spring: KShown animates toward KTarget; KPrev is the
	// value before the last change (0 until K has changed once).
	KShown  float64
	KVel    float64
	KTarget int
	KPrev   int

	// Baseline is the plain-decode tok/s from bench (0 = unknown); TokSScale
	// is the bar ceiling, which only ever grows so the bar never jumps.
	Baseline  float64
	TokSScale float64
}

// ApplyStats folds a stats snapshot in. The first snapshot snaps K into
// place; later changes leave KShown for the spring to move.
func ApplyStats(s State, stats api.Stats) State {
	next := s
	next.Stats = stats
	next.Err = ""
	if !s.Connected {
		next.KShown = float64(stats.KCurrent)
		next.KVel = 0
	}
	next.Connected = true
	if stats.KCurrent != s.KTarget {
		if s.Connected {
			next.KPrev = s.KTarget
		}
		next.KTarget = stats.KCurrent
	}
	next.TokSScale = max(s.TokSScale, minTokSScale, s.Baseline)
	if stats.TokSRecent != nil {
		next.TokSScale = max(next.TokSScale, *stats.TokSRecent)
	}
	return next
}

// ApplyChunk folds one chat stream event in. Done commits the pending
// text; an error commits whatever arrived and surfaces the message.
func ApplyChunk(s State, ev api.ChatEvent) State {
	next := s
	next.Pending = s.Pending + ev.Text
	if ev.Err != "" {
		next.Err = ev.Err
		return commitPending(next)
	}
	if ev.Done {
		return commitPending(next)
	}
	return next
}

func commitPending(s State) State {
	next := s
	next.Streaming = false
	next.Pending = ""
	if s.Pending != "" {
		next.Transcript = appendMessage(s.Transcript, api.Message{Role: "assistant", Content: s.Pending})
	}
	return next
}

// Submit queues a user turn. It is refused (false) for blank text or while
// a turn is already streaming; being disconnected does not refuse it, the
// request itself will report the failure.
func Submit(s State, text string) (State, bool) {
	text = strings.TrimSpace(text)
	if text == "" || s.Streaming {
		return s, false
	}
	next := s
	next.Transcript = appendMessage(s.Transcript, api.Message{Role: "user", Content: text})
	next.Streaming = true
	next.Pending = ""
	next.Err = ""
	return next, true
}

// StepSpring advances KShown one frame; animating is false once settled,
// at which point KShown is pinned exactly on the target.
func StepSpring(s State, a Animator) (State, bool) {
	next := s
	target := float64(s.KTarget)
	next.KShown, next.KVel = a.Step(s.KShown, s.KVel, target)
	if a.Settled(next.KShown, next.KVel, target) {
		next.KShown, next.KVel = target, 0
		return next, false
	}
	return next, true
}

// StatusLine is the header's second line.
func StatusLine(s State) string {
	status := statusConnecting
	switch {
	case s.Err != "":
		status = "error: " + s.Err
	case !s.Connected:
	case s.Streaming:
		status = statusGenerating
	case s.Stats.Busy:
		status = statusBusyOther
	default:
		status = statusIdle
	}
	return fmt.Sprintf("draft window #%d · %s", s.Stats.WindowsTotal, status)
}

// appendMessage copies before appending so the caller's slice is untouched.
func appendMessage(msgs []api.Message, m api.Message) []api.Message {
	out := make([]api.Message, len(msgs), len(msgs)+1)
	copy(out, msgs)
	return append(out, m)
}
