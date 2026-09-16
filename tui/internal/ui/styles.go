// Package ui is the Bubble Tea v2 model for `acceptrate chat`: the live
// instrument panel (window strip, gauges, springing draft depth) above a
// chat transcript. Rendering helpers are pure so they can be tested without
// a terminal.
package ui

import (
	"charm.land/lipgloss/v2"

	"github.com/kidus-der/acceptrate/tui/theme"
)

// Styles are the Lip Gloss styles resolved from one Instrument palette.
type Styles struct {
	Title     lipgloss.Style
	Muted     lipgloss.Style
	Ink       lipgloss.Style
	Label     lipgloss.Style
	Value     lipgloss.Style
	Accepted  lipgloss.Style
	Rejected  lipgloss.Style
	Bonus     lipgloss.Style
	Filled    lipgloss.Style
	Empty     lipgloss.Style
	Ghost     lipgloss.Style
	Cost      lipgloss.Style
	Code      lipgloss.Style
	User      lipgloss.Style
	Assistant lipgloss.Style
	Error     lipgloss.Style
	Rule      lipgloss.Style
	Prompt    lipgloss.Style
}

// NewStyles maps the palette onto the panel's roles. Measurement (teal) is
// for the bars, cost (amber) for the adaptation note, accepted/rejected for
// the strip alongside their glyphs.
func NewStyles(p theme.Palette) Styles {
	base := lipgloss.NewStyle()
	return Styles{
		Title:     base.Foreground(p.Ink).Bold(true),
		Muted:     base.Foreground(p.Muted),
		Ink:       base.Foreground(p.Ink),
		Label:     base.Foreground(p.Ink2),
		Value:     base.Foreground(p.Ink).Bold(true),
		Accepted:  base.Foreground(p.Accepted),
		Rejected:  base.Foreground(p.Rejected),
		Bonus:     base.Foreground(p.Measurement),
		Filled:    base.Foreground(p.Measurement),
		Empty:     base.Foreground(p.Line2),
		Ghost:     base.Foreground(p.Muted),
		Cost:      base.Foreground(p.Cost),
		Code:      base.Foreground(p.Ink2).Background(p.Surface2),
		User:      base.Foreground(p.Teal).Bold(true),
		Assistant: base.Foreground(p.Ink),
		Error:     base.Foreground(p.Red),
		Rule:      base.Foreground(p.Line),
		Prompt:    base.Foreground(p.Teal),
	}
}
