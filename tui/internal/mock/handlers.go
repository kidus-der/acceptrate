package mock

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"time"

	"github.com/kidus-der/acceptrate/tui/internal/api"
)

const (
	completionID = "chatcmpl-mock000000000000000000"
	sseMediaType = "text/event-stream"
	doneLine     = "data: [DONE]\n\n"
	busyMessage  = "a generation is already running on the single local GPU; retry shortly"
)

type chatRequest struct {
	Messages []api.Message `json:"messages"`
	Stream   bool          `json:"stream"`
}

type delta struct {
	Role    string  `json:"role,omitempty"`
	Content *string `json:"content,omitempty"`
}

type chunk struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Index        int     `json:"index"`
		Delta        delta   `json:"delta"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`
}

func (s *Server) chunkFrame(d delta, finish *string) string {
	c := chunk{ID: completionID, Object: "chat.completion.chunk", Created: time.Now().Unix(),
		Model: s.scenario.Idle.Model}
	c.Choices = append(c.Choices, struct {
		Index        int     `json:"index"`
		Delta        delta   `json:"delta"`
		FinishReason *string `json:"finish_reason"`
	}{0, d, finish})
	b, _ := json.Marshal(c)
	return "data: " + string(b) + "\n\n"
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, kind, message string) {
	writeJSON(w, status, map[string]any{"error": map[string]any{
		"message": message, "type": kind, "code": status}})
}

func (s *Server) handleChat(w http.ResponseWriter, r *http.Request) {
	var req chatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil || len(req.Messages) == 0 {
		writeJSON(w, 422, map[string]any{"detail": "messages must be a non-empty list"})
		return
	}
	if !s.state.tryAcquire() {
		writeError(w, 429, "server_busy", busyMessage)
		return
	}
	defer s.state.publish(s.scenario.Idle)
	if !req.Stream {
		s.replay(r, nil)
		s.writeCompletion(w)
		return
	}
	w.Header().Set("Content-Type", sseMediaType)
	w.Header().Set("Cache-Control", "no-cache")
	flusher, _ := w.(http.Flusher)
	emit := func(frame string) {
		_, _ = fmt.Fprint(w, frame)
		if flusher != nil {
			flusher.Flush()
		}
	}
	empty := ""
	emit(s.chunkFrame(delta{Role: "assistant", Content: &empty}, nil))
	s.replay(r, func(text string) { emit(s.chunkFrame(delta{Content: &text}, nil)) })
	stop := "stop"
	emit(s.chunkFrame(delta{}, &stop))
	emit(doneLine)
}

// replay walks the scenario, publishing each step's stats and handing its
// text to onText (nil to drain silently). It stops early if the client left.
func (s *Server) replay(r *http.Request, onText func(string)) {
	for _, step := range s.scenario.Steps {
		if s.opts.StepDelay > 0 {
			select {
			case <-time.After(s.opts.StepDelay):
			case <-r.Context().Done():
				return
			}
		}
		s.state.publish(step.Stats)
		if onText != nil {
			onText(step.Text)
		}
	}
}

func (s *Server) writeCompletion(w http.ResponseWriter) {
	text := s.scenario.Text()
	writeJSON(w, 200, map[string]any{
		"id": completionID, "object": "chat.completion", "created": time.Now().Unix(),
		"model": s.scenario.Idle.Model,
		"choices": []map[string]any{{"index": 0, "finish_reason": "stop",
			"message": api.Message{Role: "assistant", Content: text}}},
		"usage": map[string]int{"prompt_tokens": 0, "completion_tokens": len(s.scenario.Steps),
			"total_tokens": len(s.scenario.Steps)},
	})
}

func (s *Server) handleModels(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, 200, map[string]any{"object": "list", "data": []map[string]any{
		{"id": s.scenario.Idle.Model, "object": "model", "created": 0, "owned_by": "acceptrate"}}})
}

func (s *Server) handleStats(w http.ResponseWriter, _ *http.Request) {
	stats, _ := s.state.snapshot()
	writeJSON(w, 200, stats)
}

func (s *Server) handleHealthz(w http.ResponseWriter, _ *http.Request) {
	stats, _ := s.state.snapshot()
	w.Header().Set("Content-Type", "application/json")
	_, _ = fmt.Fprintf(w, `{"status":"ok","busy":%t}`, stats.Busy)
}

// handleStatsStream mirrors serve/sse.py stats_events: current stats at once,
// then `stats` on every change, else `heartbeat` every Heartbeat.
func (s *Server) handleStatsStream(w http.ResponseWriter, r *http.Request) {
	limit := -1
	if raw := r.URL.Query().Get("limit"); raw != "" {
		n, err := strconv.Atoi(raw)
		if err != nil || n < 1 {
			writeJSON(w, 422, map[string]any{"detail": "limit must be >= 1"})
			return
		}
		limit = n
	}
	w.Header().Set("Content-Type", sseMediaType)
	w.Header().Set("Cache-Control", "no-cache")
	flusher, _ := w.(http.Flusher)
	stats, version := s.state.snapshot()
	sent := 0
	emit := func(name string, st api.Stats) {
		b, _ := json.Marshal(st)
		_, _ = fmt.Fprintf(w, "event: %s\ndata: %s\n\n", name, b)
		if flusher != nil {
			flusher.Flush()
		}
		sent++
	}
	emit("stats", stats)
	for (limit < 0 || sent < limit) && r.Context().Err() == nil {
		latest := s.state.waitForChange(r.Context(), version, s.opts.Heartbeat)
		name := "heartbeat"
		if latest != version {
			name = "stats"
		}
		version = latest
		stats, _ = s.state.snapshot()
		emit(name, stats)
	}
}
