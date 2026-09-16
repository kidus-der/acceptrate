package ui

import (
	"time"

	"github.com/kidus-der/acceptrate/tui/internal/api"
)

// StatsMsg is one /stats/stream frame; Name is "stats" or "heartbeat".
type StatsMsg struct {
	Name  string
	Stats api.Stats
}

// StatsErrMsg says the stats stream dropped; the reader reconnects itself.
type StatsErrMsg struct {
	Err error
}

// ChunkMsg is one decoded chat stream frame.
type ChunkMsg struct {
	Event api.ChatEvent
}

// ChatErrMsg says the chat request failed outside the stream (HTTP error,
// connection drop, stream ended without [DONE]).
type ChatErrMsg struct {
	Err error
}

// TickMsg is one animation frame while the draft-depth spring is moving.
type TickMsg time.Time
