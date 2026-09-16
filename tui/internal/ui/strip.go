package ui

import (
	"strings"

	"charm.land/lipgloss/v2"

	"github.com/kidus-der/acceptrate/tui/internal/api"
	"github.com/kidus-der/acceptrate/tui/theme"
)

// CellKind classifies one token in the window strip.
type CellKind uint8

const (
	// CellAccepted is a draft token the target agreed with.
	CellAccepted CellKind = iota
	// CellRejected is the rejection point: the target's correction.
	CellRejected
	// CellBonus is the free token after a fully accepted window.
	CellBonus
)

// Cell is one strip position.
type Cell struct {
	Kind CellKind
}

// Glyph is the theme glyph for the cell, so the strip reads without colour.
func (c Cell) Glyph() string {
	switch c.Kind {
	case CellRejected:
		return theme.GlyphRejected
	case CellBonus:
		return theme.GlyphBonus
	default:
		return theme.GlyphAccepted
	}
}

// Cells expands recent windows into tokens: n accepted glyphs, then either
// the rejection point or the bonus token. Tokens drafted past the rejection
// were discarded and are not shown. Baseline windows (k = 0) add nothing.
func Cells(windows []api.WindowSummary) []Cell {
	var cells []Cell
	for _, w := range windows {
		if w.KProposed <= 0 {
			continue
		}
		for i := 0; i < w.NAccepted; i++ {
			cells = append(cells, Cell{Kind: CellAccepted})
		}
		if w.NAccepted < w.KProposed {
			cells = append(cells, Cell{Kind: CellRejected})
		} else {
			cells = append(cells, Cell{Kind: CellBonus})
		}
	}
	return cells
}

// RenderStrip renders the last width cells, left-aligned and padded.
func RenderStrip(cells []Cell, width int, st Styles) string {
	if width <= 0 {
		return ""
	}
	if len(cells) > width {
		cells = cells[len(cells)-width:]
	}
	var b strings.Builder
	for _, c := range cells {
		b.WriteString(cellStyle(c, st).Render(c.Glyph()))
	}
	b.WriteString(strings.Repeat(" ", width-len(cells)))
	return b.String()
}

func cellStyle(c Cell, st Styles) lipgloss.Style {
	switch c.Kind {
	case CellRejected:
		return st.Rejected
	case CellBonus:
		return st.Bonus
	default:
		return st.Accepted
	}
}
