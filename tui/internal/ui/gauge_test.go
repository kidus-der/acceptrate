package ui

import (
	"strings"
	"testing"

	"github.com/charmbracelet/x/ansi"

	"github.com/kidus-der/acceptrate/tui/theme"
)

func TestRenderBar(t *testing.T) {
	st := NewStyles(theme.Dark())
	tests := []struct {
		name  string
		frac  float64
		width int
		ghost float64 // -1 for none
		want  string
	}{
		{"empty", 0, 5, -1, "░░░░░"},
		{"half", 0.5, 4, -1, "██░░"},
		{"full", 1, 3, -1, "███"},
		{"clamped above one", 1.7, 3, -1, "███"},
		{"clamped below zero", -0.2, 3, -1, "░░░"},
		{"ghost marker on the empty part", 0.25, 8, 0.75, "██░░░░┆░"},
		{"ghost marker inside the filled part", 0.75, 8, 0.25, "██┆█████"},
		{"zero width", 0.5, 0, -1, ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ansi.Strip(RenderBar(tt.frac, tt.ghost, tt.width, st))
			if got != tt.want {
				t.Errorf("got %q, want %q", got, tt.want)
			}
		})
	}
}

func f(v float64) *float64 { return &v }

func TestAcceptanceGauge(t *testing.T) {
	st := NewStyles(theme.Dark())

	with := ansi.Strip(AcceptanceGauge(f(0.72), 10, st))
	without := ansi.Strip(AcceptanceGauge(nil, 10, st))

	if !strings.HasPrefix(with, "acceptance ") || !strings.HasSuffix(with, " 0.72") {
		t.Errorf("with alpha: %q", with)
	}
	if !strings.Contains(with, "███████░░░") {
		t.Errorf("bar for 0.72 over 10 cells: %q", with)
	}
	if !strings.HasSuffix(without, " —") || strings.Contains(without, "█") {
		t.Errorf("without alpha should show an em dash and an empty bar: %q", without)
	}
}

func TestThroughputGauge(t *testing.T) {
	st := NewStyles(theme.Dark())

	got := ansi.Strip(ThroughputGauge(f(28.4), 40, 16.6, 10, st))
	noBase := ansi.Strip(ThroughputGauge(f(28.4), 40, 0, 10, st))
	none := ansi.Strip(ThroughputGauge(nil, 40, 16.6, 10, st))

	if !strings.HasPrefix(got, "throughput ") || !strings.Contains(got, "28.4 tok/s") {
		t.Errorf("throughput: %q", got)
	}
	if !strings.Contains(got, "1.71×") || !strings.Contains(got, "┆") {
		t.Errorf("with a baseline the gauge shows the ghost marker and speedup: %q", got)
	}
	if strings.Contains(noBase, "×") || strings.Contains(noBase, "┆") {
		t.Errorf("without a baseline nothing is faked: %q", noBase)
	}
	if !strings.HasSuffix(none, " —") {
		t.Errorf("nil tok/s: %q", none)
	}
}

func TestDepthGauge(t *testing.T) {
	st := NewStyles(theme.Dark())

	adapting := ansi.Strip(DepthGauge(5.3, 6, 4, 8, 8, st))
	settled := ansi.Strip(DepthGauge(6, 6, 6, 8, 8, st))
	first := ansi.Strip(DepthGauge(4, 4, 0, 8, 8, st))

	if !strings.HasPrefix(adapting, "draft depth ") || !strings.Contains(adapting, "K = 6") {
		t.Errorf("adapting: %q", adapting)
	}
	if !strings.Contains(adapting, "↑ adapting from 4") {
		t.Errorf("adapting should name the previous K: %q", adapting)
	}
	if strings.Contains(settled, "adapting") || strings.Contains(first, "adapting") {
		t.Errorf("no adaptation note when K did not change: %q / %q", settled, first)
	}
	// 5.3 of 8 over 8 cells rounds to 5 filled.
	if !strings.Contains(adapting, "█████░░░") {
		t.Errorf("animated bar for 5.3/8: %q", adapting)
	}
}
