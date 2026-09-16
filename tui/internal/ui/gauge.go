package ui

import (
	"fmt"
	"math"
	"strings"
)

const (
	barFilled   = "█"
	barEmpty    = "░"
	barGhost    = "┆"
	noValue     = "—"
	labelWidth  = 11 // "draft depth"
	arrowUp     = "↑"
	arrowDown   = "↓"
	speedupUnit = "×"
)

// RenderBar draws frac (0–1) of width cells filled. ghost (0–1) places a
// baseline marker; pass a negative ghost for none.
func RenderBar(frac, ghost float64, width int, st Styles) string {
	if width <= 0 {
		return ""
	}
	filled := int(math.Round(clamp01(frac) * float64(width)))
	ghostAt := -1
	if ghost >= 0 {
		ghostAt = min(width-1, int(math.Round(clamp01(ghost)*float64(width))))
	}
	var b strings.Builder
	for i := 0; i < width; i++ {
		switch {
		case i == ghostAt:
			b.WriteString(st.Ghost.Render(barGhost))
		case i < filled:
			b.WriteString(st.Filled.Render(barFilled))
		default:
			b.WriteString(st.Empty.Render(barEmpty))
		}
	}
	return b.String()
}

// gaugeRow lays out "label  bar  value".
func gaugeRow(label, bar, value string, st Styles) string {
	padded := fmt.Sprintf("%-*s", labelWidth, label)
	return st.Label.Render(padded) + " " + bar + " " + value
}

// AcceptanceGauge is the 0–1 acceptance-rate bar with its value.
func AcceptanceGauge(alpha *float64, width int, st Styles) string {
	if alpha == nil {
		return gaugeRow("acceptance", RenderBar(0, -1, width, st), st.Muted.Render(noValue), st)
	}
	value := st.Value.Render(fmt.Sprintf("%.2f", *alpha))
	return gaugeRow("acceptance", RenderBar(*alpha, -1, width, st), value, st)
}

// ThroughputGauge scales tok/s against scale, ghosts the baseline when one
// is known (> 0) and then also prints the speedup. Nothing is faked when
// the baseline is unknown.
func ThroughputGauge(tokS *float64, scale, baseline float64, width int, st Styles) string {
	ghost := -1.0
	if baseline > 0 && scale > 0 {
		ghost = baseline / scale
	}
	if tokS == nil {
		return gaugeRow("throughput", RenderBar(0, ghost, width, st), st.Muted.Render(noValue), st)
	}
	frac := 0.0
	if scale > 0 {
		frac = *tokS / scale
	}
	value := st.Value.Render(fmt.Sprintf("%.1f tok/s", *tokS))
	if baseline > 0 {
		value += st.Muted.Render(fmt.Sprintf("  %.2f%s vs baseline", *tokS/baseline, speedupUnit))
	}
	return gaugeRow("throughput", RenderBar(frac, ghost, width, st), value, st)
}

// DepthGauge shows the springing K (shown) against kMax, the target K, and
// the adaptation note while K differs from the previous value.
func DepthGauge(shown float64, target, prev, kMax, width int, st Styles) string {
	frac := 0.0
	if kMax > 0 {
		frac = shown / float64(kMax)
	}
	value := st.Value.Render(fmt.Sprintf("K = %d", target))
	if prev > 0 && prev != target {
		arrow := arrowUp
		if target < prev {
			arrow = arrowDown
		}
		value += st.Cost.Render(fmt.Sprintf("  %s adapting from %d", arrow, prev))
	}
	return gaugeRow("draft depth", RenderBar(frac, -1, width, st), value, st)
}

func clamp01(v float64) float64 {
	return math.Max(0, math.Min(1, v))
}
