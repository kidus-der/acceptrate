package mock

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync"
	"time"

	"github.com/kidus-der/acceptrate/tui/internal/api"
)

// DefaultHeartbeat mirrors serve/sse.py HEARTBEAT_S.
const DefaultHeartbeat = time.Second

// Options tune the replay; zero values mean "no delay" and DefaultHeartbeat.
type Options struct {
	StepDelay time.Duration // pause before each scripted window
	Heartbeat time.Duration // idle interval on /stats/stream
}

// Server is a running mock; Close it when done.
type Server struct {
	*httptest.Server
	scenario Scenario
	opts     Options
	state    *state
}

// New starts a mock server replaying sc.
func New(sc Scenario, opts Options) *Server {
	if opts.Heartbeat <= 0 {
		opts.Heartbeat = DefaultHeartbeat
	}
	s := &Server{scenario: sc, opts: opts, state: newState(sc.Idle)}
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", s.handleChat)
	mux.HandleFunc("GET /v1/models", s.handleModels)
	mux.HandleFunc("GET /stats", s.handleStats)
	mux.HandleFunc("GET /stats/stream", s.handleStatsStream)
	mux.HandleFunc("GET /healthz", s.handleHealthz)
	s.Server = httptest.NewServer(mux)
	return s
}

// state is the one-generation-at-a-time session, versioned so the stats
// stream can wake on change like serve/session.py wait_for_change.
type state struct {
	mu      sync.Mutex
	stats   api.Stats
	version int
	changed chan struct{}
}

func newState(initial api.Stats) *state {
	return &state{stats: initial, changed: make(chan struct{})}
}

// snapshot returns the current stats and version.
func (s *state) snapshot() (api.Stats, int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.stats, s.version
}

// publish replaces the stats and wakes every waiter.
func (s *state) publish(stats api.Stats) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.stats = stats
	s.version++
	close(s.changed)
	s.changed = make(chan struct{})
}

// tryAcquire flips busy on; false if a generation already runs (-> 429).
func (s *state) tryAcquire() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.stats.Busy {
		return false
	}
	s.stats.Busy = true
	s.version++
	close(s.changed)
	s.changed = make(chan struct{})
	return true
}

// waitForChange blocks until the version moves past seen, the timeout
// elapses, or ctx ends; it returns the latest version.
func (s *state) waitForChange(ctx context.Context, seen int, timeout time.Duration) int {
	s.mu.Lock()
	ch, version := s.changed, s.version
	s.mu.Unlock()
	if version != seen {
		return version
	}
	timer := time.NewTimer(timeout)
	defer timer.Stop()
	select {
	case <-ch:
	case <-timer.C:
	case <-ctx.Done():
	}
	_, latest := s.snapshot()
	return latest
}
