// Package mock is an in-process stand-in for `acceptrate serve` that replays
// a scripted generation: no model, no MLX, just the same HTTP shapes. It
// backs the TUI's tests, the fps benchmark and `acceptrate-tui --mock`.
package mock

import (
	"math"
	"strings"

	"github.com/kidus-der/acceptrate/tui/internal/api"
)

// Step is one scripted draft window: the stats snapshot after it and the
// text it committed.
type Step struct {
	Stats api.Stats
	Text  string
}

// Scenario is a whole scripted generation plus the stats shown when idle.
type Scenario struct {
	Steps []Step
	Idle  api.Stats
}

// Text is the full assistant answer the scenario streams.
func (s Scenario) Text() string {
	var b strings.Builder
	for _, step := range s.Steps {
		b.WriteString(step.Text)
	}
	return b.String()
}

const (
	demoModel = "mlx-community/Llama-3.1-8B-Instruct-4bit"
	demoDraft = "mlx-community/Llama-3.2-1B-Instruct-4bit"

	demoAlphaStart = 0.45
	demoAlphaEnd   = 0.80
	demoTokSStart  = 18.0
	demoTokSEnd    = 31.0
	demoKLow       = 4
	demoKHigh      = 6
	demoKStepAt    = 0.4 // fraction of the way through when K adapts
	demoWordsPer   = 2   // words streamed per window
	lastWindows    = 16  // mirrors serve/stats.py LAST_WINDOWS
	msPerS         = 1000.0
)

// demoAnswer flows from prose into a code block: the brief's money shot.
const demoAnswer = "Speculative decoding pays off when the draft model agrees with the target " +
	"often enough that verifying K tokens in one batched pass costs less than K sequential " +
	"decode steps. In prose the acceptance rate drifts; in code it climbs, because the " +
	"syntax is predictable:\n\n```python\ndef adaptive_k(alpha: float, k_max: int = 8) -> int:\n" +
	"    expected = (1 - alpha ** (k_max + 1)) / (1 - alpha)\n" +
	"    return max(1, min(k_max, round(expected)))\n```\n\n" +
	"That is why the draft depth springs from 4 to 6 as the block streams."

// Demo is the brief's scenario: alpha climbs 0.45 -> 0.80 while K steps 4 -> 6
// and throughput pulls away from the baseline.
func Demo() Scenario {
	chunks := chunkWords(demoAnswer, demoWordsPer)
	n := len(chunks)
	steps := make([]Step, 0, n)
	var windows []api.WindowSummary
	accepted, proposed := 0, 0
	for i, text := range chunks {
		frac := float64(i) / float64(n-1)
		alpha := demoAlphaStart + (demoAlphaEnd-demoAlphaStart)*frac
		tokS := demoTokSStart + (demoTokSEnd-demoTokSStart)*frac
		k := demoKLow
		if frac >= demoKStepAt {
			k = demoKHigh
		}
		w := demoWindow(i, k, alpha, tokS)
		windows = append(windows, w)
		accepted += w.NAccepted
		proposed += w.KProposed
		steps = append(steps, Step{Text: text, Stats: api.Stats{
			Model: demoModel, Draft: demoDraft, Busy: true, KCurrent: k,
			AlphaEWMA: ptr(alpha), TokSRecent: ptr(tokS),
			WindowsTotal: i + 1, AcceptedTotal: accepted, ProposedTotal: proposed,
			LastWindows: tail(windows, lastWindows),
		}})
	}
	idle := steps[n-1].Stats
	idle.Busy = false
	return Scenario{Steps: steps, Idle: idle}
}

// demoWindow fabricates one window whose acceptance count wobbles around
// alpha*k so the strip shows real rejection points, not a flat bar.
func demoWindow(i, k int, alpha, tokS float64) api.WindowSummary {
	wobble := 0.5 * math.Sin(float64(i)*1.7)
	nAccepted := int(math.Round(alpha*float64(k) + wobble))
	nAccepted = max(0, min(k, nAccepted))
	committed := float64(nAccepted + 1) // the corrected token or the bonus token
	windowMs := committed / tokS * msPerS
	return api.WindowSummary{
		KProposed: k, NAccepted: nAccepted,
		DraftMs: windowMs * 0.15, VerifyMs: windowMs * 0.75, WindowMs: windowMs,
	}
}

// chunkWords splits text into groups of n words, keeping the whitespace
// (including newlines) attached to the preceding word so the join is exact.
func chunkWords(text string, n int) []string {
	var words []string
	start := 0
	for i := 0; i < len(text); i++ {
		if text[i] == ' ' && i+1 < len(text) && text[i+1] != ' ' && text[i+1] != '\n' {
			words = append(words, text[start:i+1])
			start = i + 1
		}
	}
	words = append(words, text[start:])
	var chunks []string
	for i := 0; i < len(words); i += n {
		chunks = append(chunks, strings.Join(words[i:min(i+n, len(words))], ""))
	}
	return chunks
}

func tail(w []api.WindowSummary, n int) []api.WindowSummary {
	if len(w) > n {
		w = w[len(w)-n:]
	}
	out := make([]api.WindowSummary, len(w))
	copy(out, w)
	return out
}

func ptr(v float64) *float64 { return &v }
