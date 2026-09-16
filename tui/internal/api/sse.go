// Package api is the HTTP client for `acceptrate serve`: the OpenAI-compatible
// chat stream and the /stats/stream feed. Parsers are pure functions so the
// frame shapes can be table-tested against the ones serve/sse.py emits.
package api

import (
	"bufio"
	"io"
	"strings"
)

// Frame is one server-sent event. Event is empty for OpenAI chat chunks and
// "stats" / "heartbeat" on the stats feed.
type Frame struct {
	Event string
	Data  string
}

// maxLineBytes bounds a single SSE line; a stats payload is a few KB.
const maxLineBytes = 1 << 20

// ReadFrames parses SSE frames from r and calls fn for each one. It returns
// fn's error as soon as it is non-nil, otherwise the reader's error (nil on
// a clean EOF). An unterminated trailing frame is delivered.
func ReadFrames(r io.Reader, fn func(Frame) error) error {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, 64*1024), maxLineBytes)
	var event string
	var data []string
	flush := func() error {
		if event == "" && len(data) == 0 {
			return nil
		}
		frame := Frame{Event: event, Data: strings.Join(data, "\n")}
		event, data = "", nil
		return fn(frame)
	}
	for scanner.Scan() {
		line := strings.TrimSuffix(scanner.Text(), "\r")
		if line == "" {
			if err := flush(); err != nil {
				return err
			}
			continue
		}
		event, data = parseLine(line, event, data)
	}
	if err := scanner.Err(); err != nil {
		return err
	}
	return flush()
}

// parseLine folds one SSE line into the pending (event, data) accumulator.
func parseLine(line, event string, data []string) (string, []string) {
	switch {
	case strings.HasPrefix(line, ":"):
		return event, data
	case strings.HasPrefix(line, "event:"):
		return strings.TrimSpace(strings.TrimPrefix(line, "event:")), data
	case strings.HasPrefix(line, "data:"):
		return event, append(data, strings.TrimPrefix(strings.TrimPrefix(line, "data:"), " "))
	}
	return event, data
}
