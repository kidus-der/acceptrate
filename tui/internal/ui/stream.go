package ui

import (
	"context"
	"errors"
	"time"

	tea "charm.land/bubbletea/v2"

	"github.com/kidus-der/acceptrate/tui/internal/api"
)

// reconnectDelay is the pause before re-opening a dropped stats stream.
const reconnectDelay = time.Second

var errNoDone = errors.New("stream ended without [DONE]")

// runStats follows /stats/stream for the life of ctx, reconnecting after
// every drop and reporting each failure once.
func runStats(ctx context.Context, client *api.Client, ch chan<- tea.Msg) {
	for {
		err := client.StreamStats(ctx, func(name string, s api.Stats) error {
			return send(ctx, ch, StatsMsg{Name: name, Stats: s})
		})
		if ctx.Err() != nil {
			return
		}
		if err != nil {
			if send(ctx, ch, StatsErrMsg{Err: err}) != nil {
				return
			}
		}
		if !sleep(ctx, reconnectDelay) {
			return
		}
	}
}

// runChat streams one turn; a stream that ends without [DONE] is an error
// so the model never stays stuck in the generating state.
func runChat(ctx context.Context, client *api.Client, msgs []api.Message, ch chan<- tea.Msg) {
	done := false
	err := client.StreamChat(ctx, msgs, func(ev api.ChatEvent) error {
		done = done || ev.Done
		return send(ctx, ch, ChunkMsg{Event: ev})
	})
	if ctx.Err() != nil {
		return
	}
	if err == nil && !done {
		err = errNoDone
	}
	if err != nil {
		_ = send(ctx, ch, ChatErrMsg{Err: err})
	}
}

func send(ctx context.Context, ch chan<- tea.Msg, msg tea.Msg) error {
	select {
	case ch <- msg:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func sleep(ctx context.Context, d time.Duration) bool {
	timer := time.NewTimer(d)
	defer timer.Stop()
	select {
	case <-timer.C:
		return true
	case <-ctx.Done():
		return false
	}
}
