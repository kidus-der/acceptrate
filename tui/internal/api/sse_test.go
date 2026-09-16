package api

import (
	"errors"
	"strings"
	"testing"
)

// Frames are copied from the shapes serve/sse.py emits and tests/test_serve_app.py asserts.
const (
	statsFrame     = "event: stats\ndata: {\"model\":\"target-fake\"}\n\n"
	heartbeatFrame = "event: heartbeat\ndata: {\"model\":\"target-fake\"}\n\n"
	chunkFrame     = "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\"}\n\n"
	doneFrame      = "data: [DONE]\n\n"
)

func TestReadFramesSplitsEventAndData(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  []Frame
	}{
		{
			name:  "named stats event",
			input: statsFrame,
			want:  []Frame{{Event: "stats", Data: `{"model":"target-fake"}`}},
		},
		{
			name:  "unnamed openai chunk then done",
			input: chunkFrame + doneFrame,
			want: []Frame{
				{Event: "", Data: `{"id":"chatcmpl-1","object":"chat.completion.chunk"}`},
				{Event: "", Data: "[DONE]"},
			},
		},
		{
			name:  "stats then heartbeats",
			input: statsFrame + heartbeatFrame + heartbeatFrame,
			want: []Frame{
				{Event: "stats", Data: `{"model":"target-fake"}`},
				{Event: "heartbeat", Data: `{"model":"target-fake"}`},
				{Event: "heartbeat", Data: `{"model":"target-fake"}`},
			},
		},
		{
			name:  "multi-line data joins with newline",
			input: "data: a\ndata: b\n\n",
			want:  []Frame{{Data: "a\nb"}},
		},
		{
			name:  "comments and blank leading lines are ignored",
			input: ":keepalive\n\n\ndata: x\n\n",
			want:  []Frame{{Data: "x"}},
		},
		{
			name:  "crlf line endings",
			input: "event: stats\r\ndata: y\r\n\r\n",
			want:  []Frame{{Event: "stats", Data: "y"}},
		},
		{
			name:  "unterminated final frame is still delivered",
			input: "data: tail",
			want:  []Frame{{Data: "tail"}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var got []Frame
			err := ReadFrames(strings.NewReader(tt.input), func(f Frame) error {
				got = append(got, f)
				return nil
			})
			if err != nil {
				t.Fatalf("ReadFrames: %v", err)
			}
			if len(got) != len(tt.want) {
				t.Fatalf("got %d frames %+v, want %d %+v", len(got), got, len(tt.want), tt.want)
			}
			for i := range got {
				if got[i] != tt.want[i] {
					t.Errorf("frame %d: got %+v, want %+v", i, got[i], tt.want[i])
				}
			}
		})
	}
}

func TestReadFramesStopsWhenCallbackErrors(t *testing.T) {
	stop := errors.New("stop")
	calls := 0

	err := ReadFrames(strings.NewReader(statsFrame+heartbeatFrame), func(Frame) error {
		calls++
		return stop
	})

	if !errors.Is(err, stop) {
		t.Fatalf("err = %v, want %v", err, stop)
	}
	if calls != 1 {
		t.Fatalf("callback called %d times, want 1", calls)
	}
}
