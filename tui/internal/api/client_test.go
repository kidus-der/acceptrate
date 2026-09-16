package api_test

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/kidus-der/acceptrate/tui/internal/api"
	"github.com/kidus-der/acceptrate/tui/internal/mock"
)

var fast = mock.Options{Heartbeat: 10 * time.Millisecond}

func TestStreamChatDeliversTextThenDone(t *testing.T) {
	sc := mock.Demo()
	srv := mock.New(sc, fast)
	defer srv.Close()
	client := api.New(srv.URL)

	var text string
	var done bool
	err := client.StreamChat(context.Background(), []api.Message{{Role: "user", Content: "hi"}},
		func(ev api.ChatEvent) error {
			text += ev.Text
			done = done || ev.Done
			return nil
		})

	if err != nil {
		t.Fatalf("StreamChat: %v", err)
	}
	if text != sc.Text() || !done {
		t.Errorf("text=%q done=%v", text, done)
	}
}

func TestStreamChatSurfacesHTTPErrorsAsMessages(t *testing.T) {
	srv := mock.New(mock.Demo(), fast)
	defer srv.Close()
	client := api.New(srv.URL)

	err := client.StreamChat(context.Background(), nil, func(api.ChatEvent) error { return nil })

	var httpErr *api.HTTPError
	if !errors.As(err, &httpErr) || httpErr.Status != 422 {
		t.Fatalf("err = %v, want HTTPError 422", err)
	}
}

func TestStreamStatsDeliversNamedEventsUntilCancelled(t *testing.T) {
	srv := mock.New(mock.Demo(), fast)
	defer srv.Close()
	client := api.New(srv.URL)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	var events []string
	err := client.StreamStats(ctx, func(name string, s api.Stats) error {
		events = append(events, name)
		if s.Model == "" {
			t.Errorf("stats without model: %+v", s)
		}
		if len(events) == 3 {
			cancel()
		}
		return nil
	})

	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
	if len(events) < 3 || events[0] != "stats" || events[1] != "heartbeat" {
		t.Errorf("events = %v", events)
	}
}

func TestStreamStatsStopsOnCallbackError(t *testing.T) {
	srv := mock.New(mock.Demo(), fast)
	defer srv.Close()
	stop := errors.New("stop")

	err := api.New(srv.URL).StreamStats(context.Background(),
		func(string, api.Stats) error { return stop })

	if !errors.Is(err, stop) {
		t.Fatalf("err = %v, want stop", err)
	}
}

func TestStreamStatsFailsFastWhenServerIsDown(t *testing.T) {
	srv := mock.New(mock.Demo(), fast)
	url := srv.URL
	srv.Close()

	err := api.New(url).StreamStats(context.Background(), func(string, api.Stats) error { return nil })

	if err == nil {
		t.Fatal("want a connection error")
	}
}
