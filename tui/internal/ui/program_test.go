package ui

import (
	"io"
	"testing"
	"time"

	tea "charm.land/bubbletea/v2"

	"github.com/kidus-der/acceptrate/tui/internal/api"
	"github.com/kidus-der/acceptrate/tui/internal/mock"
	"github.com/kidus-der/acceptrate/tui/theme"
)

// Runs a real headless tea.Program against the mock: the stats goroutine
// connects, typed keys submit a turn, chunks stream in, [DONE] commits.
func TestProgramStreamsATurnEndToEndAgainstTheMock(t *testing.T) {
	sc := mock.Demo()
	srv := mock.New(sc, mock.Options{Heartbeat: 20 * time.Millisecond})
	defer srv.Close()
	m := New(Config{Client: api.New(srv.URL), URL: srv.URL, Styles: NewStyles(theme.Dark()), FPS: 30})

	connected := make(chan struct{}, 1)
	finished := make(chan struct{}, 1)
	filter := func(_ tea.Model, msg tea.Msg) tea.Msg {
		switch msg := msg.(type) {
		case StatsMsg:
			select {
			case connected <- struct{}{}:
			default:
			}
		case ChunkMsg:
			if msg.Event.Done {
				select {
				case finished <- struct{}{}:
				default:
				}
			}
		}
		return msg
	}
	p := tea.NewProgram(m, tea.WithoutRenderer(), tea.WithInput(nil), tea.WithOutput(io.Discard),
		tea.WithFilter(filter))
	result := make(chan tea.Model, 1)
	go func() {
		final, err := p.Run()
		if err != nil {
			t.Errorf("Run: %v", err)
		}
		result <- final
	}()

	p.Send(tea.WindowSizeMsg{Width: 80, Height: 24})
	waitFor(t, connected, "first stats event")
	for _, r := range "hi" {
		p.Send(tea.KeyPressMsg{Code: r, Text: string(r)})
	}
	p.Send(tea.KeyPressMsg{Code: tea.KeyEnter})
	waitFor(t, finished, "[DONE]")
	p.Send(tea.KeyPressMsg{Code: 'c', Mod: tea.ModCtrl})

	final := (<-result).(Model)
	if !final.state.Connected || final.state.Streaming {
		t.Errorf("final state: connected=%v streaming=%v err=%q",
			final.state.Connected, final.state.Streaming, final.state.Err)
	}
	if n := len(final.state.Transcript); n != 2 || final.state.Transcript[1].Content != sc.Text() {
		t.Errorf("transcript (%d turns) = %+v", n, final.state.Transcript)
	}
	if final.state.Transcript[0].Content != "hi" {
		t.Errorf("user turn = %q", final.state.Transcript[0].Content)
	}
	if final.state.KTarget != 6 || final.state.KPrev != 4 {
		t.Errorf("K should have adapted 4 -> 6: target=%d prev=%d", final.state.KTarget, final.state.KPrev)
	}
}

func waitFor(t *testing.T, ch <-chan struct{}, what string) {
	t.Helper()
	select {
	case <-ch:
	case <-time.After(5 * time.Second):
		t.Fatalf("timed out waiting for %s", what)
	}
}
