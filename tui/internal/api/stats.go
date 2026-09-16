package api

import (
	"encoding/json"
	"fmt"
)

// WindowSummary is one recent draft window (serve/stats.py WindowSummary).
type WindowSummary struct {
	KProposed int     `json:"k_proposed"`
	NAccepted int     `json:"n_accepted"`
	DraftMs   float64 `json:"draft_ms"`
	VerifyMs  float64 `json:"verify_ms"`
	WindowMs  float64 `json:"window_ms"`
}

// Stats is the /stats body and the payload of every /stats/stream frame.
// AlphaEWMA and TokSRecent are nil until the first speculative window.
type Stats struct {
	Model         string          `json:"model"`
	Draft         string          `json:"draft"`
	Busy          bool            `json:"busy"`
	KCurrent      int             `json:"k_current"`
	AlphaEWMA     *float64        `json:"alpha_ewma"`
	TokSRecent    *float64        `json:"tok_s_recent"`
	WindowsTotal  int             `json:"windows_total"`
	AcceptedTotal int             `json:"accepted_total"`
	ProposedTotal int             `json:"proposed_total"`
	LastWindows   []WindowSummary `json:"last_windows"`
}

// ParseStats decodes a stats payload.
func ParseStats(data string) (Stats, error) {
	var stats Stats
	if err := json.Unmarshal([]byte(data), &stats); err != nil {
		return Stats{}, fmt.Errorf("stats: %w", err)
	}
	if stats.LastWindows == nil {
		stats.LastWindows = []WindowSummary{}
	}
	return stats, nil
}
