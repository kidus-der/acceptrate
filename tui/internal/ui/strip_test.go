package ui

import (
	"strings"
	"testing"

	"github.com/charmbracelet/x/ansi"

	"github.com/kidus-der/acceptrate/tui/internal/api"
	"github.com/kidus-der/acceptrate/tui/theme"
)

func TestCellsFromWindows(t *testing.T) {
	tests := []struct {
		name    string
		windows []api.WindowSummary
		want    string // glyph sequence
	}{
		{
			name:    "all accepted earns a bonus token",
			windows: []api.WindowSummary{{KProposed: 3, NAccepted: 3}},
			want:    "●●●◆",
		},
		{
			name:    "partial acceptance ends at the rejection point",
			windows: []api.WindowSummary{{KProposed: 3, NAccepted: 1}},
			want:    "●○",
		},
		{
			name:    "nothing accepted is a lone rejection",
			windows: []api.WindowSummary{{KProposed: 4, NAccepted: 0}},
			want:    "○",
		},
		{
			name: "windows concatenate in order",
			windows: []api.WindowSummary{
				{KProposed: 2, NAccepted: 2}, {KProposed: 2, NAccepted: 0},
			},
			want: "●●◆○",
		},
		{
			name:    "a baseline window (k=0) contributes nothing",
			windows: []api.WindowSummary{{KProposed: 0, NAccepted: 0}},
			want:    "",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var got strings.Builder
			for _, c := range Cells(tt.windows) {
				got.WriteString(c.Glyph())
			}
			if got.String() != tt.want {
				t.Errorf("got %q, want %q", got.String(), tt.want)
			}
		})
	}
}

func TestRenderStripKeepsTheLastWidthCells(t *testing.T) {
	st := NewStyles(theme.Dark())
	windows := []api.WindowSummary{
		{KProposed: 4, NAccepted: 4}, // ●●●●◆
		{KProposed: 4, NAccepted: 2}, // ●●○
	}

	got := ansi.Strip(RenderStrip(Cells(windows), 4, st))

	if got != "●●●○" {
		t.Errorf("got %q, want the last 4 cells", got)
	}
	if w := ansi.StringWidth(got); w != 4 {
		t.Errorf("width = %d, want 4", w)
	}
}

func TestRenderStripPadsShortSequences(t *testing.T) {
	st := NewStyles(theme.Dark())

	got := ansi.Strip(RenderStrip(Cells([]api.WindowSummary{{KProposed: 1, NAccepted: 1}}), 6, st))

	if got != "●◆    " {
		t.Errorf("got %q, want the cells left-aligned and padded", got)
	}
}

func TestStripUsesThemeGlyphsNotColourAlone(t *testing.T) {
	st := NewStyles(theme.Dark())
	windows := []api.WindowSummary{{KProposed: 2, NAccepted: 1}, {KProposed: 1, NAccepted: 1}}

	plain := ansi.Strip(RenderStrip(Cells(windows), 10, st))

	for _, g := range []string{theme.GlyphAccepted, theme.GlyphRejected, theme.GlyphBonus} {
		if !strings.Contains(plain, g) {
			t.Errorf("strip %q lacks glyph %q", plain, g)
		}
	}
}
