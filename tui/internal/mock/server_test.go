package mock

import (
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/kidus-der/acceptrate/tui/internal/api"
)

// fast keeps heartbeats short so the limit=3 stream returns quickly.
var fast = Options{Heartbeat: 10 * time.Millisecond}

func postChat(t *testing.T, url string, body string) *http.Response {
	t.Helper()
	resp, err := http.Post(url+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	return resp
}

func readAll(t *testing.T, resp *http.Response) string {
	t.Helper()
	defer resp.Body.Close()
	b, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	return string(b)
}

func TestDemoScenarioRampsAlphaAndStepsK(t *testing.T) {
	sc := Demo()

	if len(sc.Steps) < 20 {
		t.Fatalf("demo has %d steps, want a long enough ramp", len(sc.Steps))
	}
	first, last := sc.Steps[0].Stats, sc.Steps[len(sc.Steps)-1].Stats
	if first.KCurrent != 4 || last.KCurrent != 6 {
		t.Errorf("K %d -> %d, want 4 -> 6", first.KCurrent, last.KCurrent)
	}
	if *first.AlphaEWMA > 0.5 || *last.AlphaEWMA < 0.78 {
		t.Errorf("alpha %.2f -> %.2f, want 0.45 -> 0.8", *first.AlphaEWMA, *last.AlphaEWMA)
	}
	if !strings.Contains(sc.Text(), "```") {
		t.Errorf("demo text should flow into a code block")
	}
	for i, step := range sc.Steps {
		if !step.Stats.Busy {
			t.Errorf("step %d not busy", i)
		}
		if len(step.Stats.LastWindows) == 0 || len(step.Stats.LastWindows) > 16 {
			t.Errorf("step %d has %d last_windows", i, len(step.Stats.LastWindows))
		}
	}
	if sc.Idle.Busy || sc.Idle.KCurrent != 6 {
		t.Errorf("idle stats = %+v", sc.Idle)
	}
}

func TestStatsStreamSendsStatsFirstThenHeartbeats(t *testing.T) {
	srv := New(Demo(), fast)
	defer srv.Close()

	resp, err := http.Get(srv.URL + "/stats/stream?limit=3")
	if err != nil {
		t.Fatal(err)
	}
	body := readAll(t, resp)

	if ct := resp.Header.Get("Content-Type"); !strings.HasPrefix(ct, "text/event-stream") {
		t.Errorf("content-type = %q", ct)
	}
	var events []string
	for _, e := range strings.Split(body, "\n\n") {
		if strings.TrimSpace(e) != "" {
			events = append(events, e)
		}
	}
	if len(events) != 3 {
		t.Fatalf("got %d events: %q", len(events), body)
	}
	if !strings.HasPrefix(events[0], "event: stats\n") {
		t.Errorf("first event %q", events[0])
	}
	for _, e := range events[1:] {
		if !strings.HasPrefix(e, "event: heartbeat\n") {
			t.Errorf("event %q, want heartbeat", e)
		}
	}
}

func TestChatStreamsScenarioTextThenDoneAndAdvancesStats(t *testing.T) {
	sc := Demo()
	srv := New(sc, fast)
	defer srv.Close()

	resp := postChat(t, srv.URL, `{"messages":[{"role":"user","content":"hi"}],"stream":true}`)
	body := readAll(t, resp)

	if resp.StatusCode != 200 {
		t.Fatalf("status %d: %s", resp.StatusCode, body)
	}
	var text, finish string
	var done bool
	err := api.ReadFrames(strings.NewReader(body), func(f api.Frame) error {
		ev, err := api.ParseChatChunk(f.Data)
		if err != nil {
			return err
		}
		text += ev.Text
		if ev.Finish != "" {
			finish = ev.Finish
		}
		done = done || ev.Done
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if text != sc.Text() {
		t.Errorf("text = %q, want scenario text", text)
	}
	if finish != "stop" || !done {
		t.Errorf("finish=%q done=%v", finish, done)
	}

	statsResp, err := http.Get(srv.URL + "/stats")
	if err != nil {
		t.Fatal(err)
	}
	var stats api.Stats
	if err := json.NewDecoder(statsResp.Body).Decode(&stats); err != nil {
		t.Fatal(err)
	}
	if stats.Busy || stats.KCurrent != 6 {
		t.Errorf("stats after chat = %+v", stats)
	}
}

func TestNonStreamChatReturnsTheWholeText(t *testing.T) {
	sc := Demo()
	srv := New(sc, fast)
	defer srv.Close()

	resp := postChat(t, srv.URL, `{"messages":[{"role":"user","content":"hi"}]}`)
	var body struct {
		Choices []struct {
			Message api.Message `json:"message"`
		} `json:"choices"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatal(err)
	}

	if len(body.Choices) != 1 || body.Choices[0].Message.Content != sc.Text() {
		t.Errorf("body = %+v", body)
	}
}

func TestMalformedChatBodyIs422(t *testing.T) {
	srv := New(Demo(), fast)
	defer srv.Close()

	resp := postChat(t, srv.URL, `{"messages":[]}`)
	readAll(t, resp)

	if resp.StatusCode != 422 {
		t.Errorf("status = %d, want 422", resp.StatusCode)
	}
}

func TestModelsAndHealthz(t *testing.T) {
	srv := New(Demo(), fast)
	defer srv.Close()

	resp, err := http.Get(srv.URL + "/v1/models")
	if err != nil {
		t.Fatal(err)
	}
	if body := readAll(t, resp); !strings.Contains(body, Demo().Idle.Model) {
		t.Errorf("models body %q", body)
	}
	resp, err = http.Get(srv.URL + "/healthz")
	if err != nil {
		t.Fatal(err)
	}
	if body := readAll(t, resp); body != `{"status":"ok","busy":false}` {
		t.Errorf("healthz body %q", body)
	}
}
