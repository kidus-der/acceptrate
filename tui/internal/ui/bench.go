package ui

import (
	"io"
	"time"

	tea "charm.land/bubbletea/v2"

	"github.com/kidus-der/acceptrate/tui/internal/api"
	"github.com/kidus-der/acceptrate/tui/internal/mock"
)

// BenchConfig sizes a synthetic run. Styles is required; a zero FPS uses
// DefaultFPS.
type BenchConfig struct {
	Frames        int
	FPS           int
	Width, Height int
	Styles        Styles
}

// BenchResult is what the benchmark achieved.
type BenchResult struct {
	Frames  int
	Elapsed time.Duration
	FPS     float64
}

// Benchmark drives the model through Frames busy frames — each one folds in
// a demo stats step, a chat chunk and an animation tick, then writes the
// rendered view to w — and reports the achieved frame rate. It measures
// model update + view generation, the part this package owns; Bubble Tea's
// renderer then diffs cells on top of that.
func Benchmark(cfg BenchConfig, w io.Writer) BenchResult {
	m := New(Config{Styles: cfg.Styles, FPS: cfg.FPS, Baseline: 16.6})
	m = m.resize(cfg.Width, cfg.Height)
	steps := mock.Demo().Steps
	m.state, _ = Submit(m.state, "benchmark")
	m = m.refreshTranscript()

	start := time.Now()
	for i := 0; i < cfg.Frames; i++ {
		step := steps[i%len(steps)]
		m = apply(m, StatsMsg{Name: "stats", Stats: step.Stats})
		m = apply(m, ChunkMsg{Event: api.ChatEvent{Text: step.Text}})
		m = apply(m, TickMsg(start))
		_, _ = io.WriteString(w, m.View().Content)
	}
	elapsed := time.Since(start)
	return BenchResult{Frames: cfg.Frames, Elapsed: elapsed, FPS: float64(cfg.Frames) / elapsed.Seconds()}
}

func apply(m Model, msg tea.Msg) Model {
	next, _ := m.Update(msg)
	return next.(Model)
}
