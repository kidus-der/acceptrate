package api

import "testing"

// The /stats body serve/stats.py emits (fields asserted by test_serve_app.py).
const statsJSON = `{"model":"target-fake","draft":"draft-fake","busy":true,"k_current":3,` +
	`"alpha_ewma":0.5,"tok_s_recent":41.25,"windows_total":9,"accepted_total":4,` +
	`"proposed_total":27,"last_windows":[` +
	`{"k_proposed":3,"n_accepted":3,"draft_ms":1.5,"verify_ms":9.0,"window_ms":11.0},` +
	`{"k_proposed":3,"n_accepted":1,"draft_ms":1.5,"verify_ms":9.0,"window_ms":11.0}]}`

func TestParseStats(t *testing.T) {
	got, err := ParseStats(statsJSON)
	if err != nil {
		t.Fatalf("ParseStats: %v", err)
	}

	if got.Model != "target-fake" || got.Draft != "draft-fake" {
		t.Errorf("model/draft = %q/%q", got.Model, got.Draft)
	}
	if !got.Busy || got.KCurrent != 3 {
		t.Errorf("busy=%v k=%d", got.Busy, got.KCurrent)
	}
	if got.AlphaEWMA == nil || *got.AlphaEWMA != 0.5 {
		t.Errorf("alpha = %v", got.AlphaEWMA)
	}
	if got.TokSRecent == nil || *got.TokSRecent != 41.25 {
		t.Errorf("tok/s = %v", got.TokSRecent)
	}
	if got.WindowsTotal != 9 || got.AcceptedTotal != 4 || got.ProposedTotal != 27 {
		t.Errorf("totals = %d %d %d", got.WindowsTotal, got.AcceptedTotal, got.ProposedTotal)
	}
	if len(got.LastWindows) != 2 {
		t.Fatalf("last_windows len = %d", len(got.LastWindows))
	}
	w := got.LastWindows[1]
	if w.KProposed != 3 || w.NAccepted != 1 || w.DraftMs != 1.5 || w.VerifyMs != 9 || w.WindowMs != 11 {
		t.Errorf("window[1] = %+v", w)
	}
}

func TestParseStatsNullsAndNoDraft(t *testing.T) {
	got, err := ParseStats(`{"model":"m","draft":null,"busy":false,"k_current":0,` +
		`"alpha_ewma":null,"tok_s_recent":null,"windows_total":0,"accepted_total":0,` +
		`"proposed_total":0,"last_windows":[]}`)
	if err != nil {
		t.Fatalf("ParseStats: %v", err)
	}

	if got.Draft != "" || got.AlphaEWMA != nil || got.TokSRecent != nil {
		t.Errorf("nulls not preserved: %+v", got)
	}
	if got.LastWindows == nil {
		t.Errorf("last_windows should be an empty slice, not nil")
	}
}

func TestParseStatsRejectsGarbage(t *testing.T) {
	if _, err := ParseStats(`{"model":`); err == nil {
		t.Fatal("want error")
	}
}
