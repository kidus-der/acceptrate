package ui

import (
	"fmt"
	"strings"

	tea "charm.land/bubbletea/v2"
	"github.com/charmbracelet/x/ansi"

	"github.com/kidus-der/acceptrate/tui/internal/api"
)

const (
	// headerLines is the instrument panel: title, status, strip, three
	// gauges, footer, rule.
	headerLines = 8
	stripWidth  = 48
	barMin      = 8
	barMax      = 30
	kMax        = 8
	codeFence   = "```"
	cursorGlyph = "▌"
	titleText   = "acceptrate chat"
	userLabel   = "you"
	assistLabel = "assistant"
)

// View renders the full-screen frame.
func (m Model) View() tea.View {
	v := tea.NewView(m.render())
	v.AltScreen = true
	return v
}

func (m Model) render() string {
	if m.width <= 0 || m.height <= 0 {
		return ""
	}
	st := m.cfg.Styles
	lines := []string{
		m.title(),
		st.Muted.Render(StatusLine(m.state)),
		RenderStrip(Cells(m.state.Stats.LastWindows), min(m.width, stripWidth), st),
	}
	lines = append(lines, m.gauges()...)
	lines = append(lines, m.footer(), st.Rule.Render(strings.Repeat("─", m.width)))
	for i, line := range lines {
		lines[i] = ansi.Truncate(line, m.width, "")
	}
	lines = append(lines, m.vp.View())
	if !m.cfg.StatsOnly {
		lines = append(lines, ansi.Truncate(m.input.View(), m.width, ""))
	}
	return strings.Join(lines, "\n")
}

func (m Model) title() string {
	st := m.cfg.Styles
	head := st.Title.Render(titleText)
	if m.state.Stats.Model == "" {
		return head + st.Muted.Render(" — "+m.cfg.URL)
	}
	line := head + st.Muted.Render(" — ") + st.Ink.Render(shortName(m.state.Stats.Model))
	if m.state.Stats.Draft != "" {
		line += st.Muted.Render(" ← draft ") + st.Ink.Render(shortName(m.state.Stats.Draft))
	}
	return line
}

// shortName drops the hub org ("mlx-community/") so both names fit one line.
func shortName(repo string) string {
	if i := strings.LastIndex(repo, "/"); i >= 0 {
		return repo[i+1:]
	}
	return repo
}

func (m Model) gauges() []string {
	st := m.cfg.Styles
	bar := max(barMin, min(barMax, m.width/3))
	s := m.state
	return []string{
		AcceptanceGauge(s.Stats.AlphaEWMA, bar, st),
		ThroughputGauge(s.Stats.TokSRecent, s.TokSScale, s.Baseline, bar, st),
		DepthGauge(s.KShown, s.KTarget, s.KPrev, kMax, bar, st),
	}
}

// footer shows only what /stats provides: totals and the last window's
// numbers. Memory, thermal and page-ins are not in the feed, so they are
// omitted rather than faked.
func (m Model) footer() string {
	st := m.cfg.Styles
	s := m.state.Stats
	parts := []string{fmt.Sprintf("windows %d", s.WindowsTotal)}
	if s.ProposedTotal > 0 {
		parts = append(parts, fmt.Sprintf("accepted %d/%d", s.AcceptedTotal, s.ProposedTotal))
	}
	if n := len(s.LastWindows); n > 0 {
		last := s.LastWindows[n-1]
		parts = append(parts,
			fmt.Sprintf("last window %d/%d", last.NAccepted, last.KProposed),
			fmt.Sprintf("draft %.1f ms", last.DraftMs),
			fmt.Sprintf("verify %.1f ms", last.VerifyMs))
	}
	return st.Muted.Render(strings.Join(parts, " · "))
}

// renderTranscript lays out the turns; code fences get the code style and
// the streaming turn ends in a cursor glyph.
func renderTranscript(msgs []api.Message, pending string, streaming bool, width int, st Styles) string {
	var blocks []string
	for _, msg := range msgs {
		blocks = append(blocks, renderTurn(msg, width, st))
	}
	if streaming {
		text := pending + cursorGlyph
		blocks = append(blocks, renderTurn(api.Message{Role: "assistant", Content: text}, width, st))
	}
	return strings.Join(blocks, "\n\n")
}

func renderTurn(msg api.Message, width int, st Styles) string {
	if msg.Role == "user" {
		return st.User.Render(userLabel) + "\n" + st.Ink.Render(msg.Content)
	}
	return st.Muted.Render(assistLabel) + "\n" + renderCodeFences(msg.Content, width, st)
}

func renderCodeFences(text string, width int, st Styles) string {
	lines := strings.Split(text, "\n")
	inCode := false
	for i, line := range lines {
		if strings.HasPrefix(strings.TrimSpace(line), codeFence) {
			inCode = !inCode
			lines[i] = st.Muted.Render(line)
			continue
		}
		if inCode {
			lines[i] = st.Code.Width(max(1, width)).Render(line)
		} else {
			lines[i] = st.Assistant.Render(line)
		}
	}
	return strings.Join(lines, "\n")
}
