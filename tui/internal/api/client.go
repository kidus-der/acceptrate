package api

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// DefaultURL is where `acceptrate serve` binds by default.
const DefaultURL = "http://127.0.0.1:8321"

// DefaultMaxTokens matches serve/schemas.py DEFAULT_MAX_TOKENS.
const DefaultMaxTokens = 512

// errorBodyLimit caps how much of a failed response we keep for the message.
const errorBodyLimit = 4096

// HTTPError is a non-2xx response; Message is the server's error text when
// the body was the OpenAI envelope, else the raw body.
type HTTPError struct {
	Status  int
	Message string
}

func (e *HTTPError) Error() string {
	return fmt.Sprintf("server returned %d: %s", e.Status, e.Message)
}

// Client talks to one `acceptrate serve` instance.
type Client struct {
	BaseURL   string
	HTTP      *http.Client
	MaxTokens int
}

// New returns a client for baseURL with no request timeout (streams are
// long-lived; cancel via context instead).
func New(baseURL string) *Client {
	return &Client{
		BaseURL:   strings.TrimRight(baseURL, "/"),
		HTTP:      &http.Client{},
		MaxTokens: DefaultMaxTokens,
	}
}

// StreamChat posts messages with stream: true and calls fn per decoded
// frame until [DONE], fn errors, or the connection drops.
func (c *Client) StreamChat(ctx context.Context, msgs []Message, fn func(ChatEvent) error) error {
	body, err := json.Marshal(map[string]any{
		"messages": msgs, "stream": true, "max_tokens": c.MaxTokens, "temperature": 0,
	})
	if err != nil {
		return fmt.Errorf("encode request: %w", err)
	}
	resp, err := c.do(ctx, http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	return ReadFrames(resp.Body, func(f Frame) error {
		ev, err := ParseChatChunk(f.Data)
		if err != nil {
			return err
		}
		return fn(ev)
	})
}

// StreamStats follows /stats/stream, calling fn with the event name
// ("stats" or "heartbeat") and payload until ctx ends or fn errors.
func (c *Client) StreamStats(ctx context.Context, fn func(name string, s Stats) error) error {
	resp, err := c.do(ctx, http.MethodGet, "/stats/stream", nil)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	err = ReadFrames(resp.Body, func(f Frame) error {
		stats, err := ParseStats(f.Data)
		if err != nil {
			return err
		}
		return fn(f.Event, stats)
	})
	if ctx.Err() != nil {
		return ctx.Err()
	}
	return err
}

// Stats fetches one /stats snapshot.
func (c *Client) Stats(ctx context.Context) (Stats, error) {
	resp, err := c.do(ctx, http.MethodGet, "/stats", nil)
	if err != nil {
		return Stats{}, err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return Stats{}, err
	}
	return ParseStats(string(body))
}

// do issues a request and turns non-2xx responses into *HTTPError.
func (c *Client) do(ctx context.Context, method, path string, body io.Reader) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, method, c.BaseURL+path, body)
	if err != nil {
		return nil, err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return nil, fmt.Errorf("connect to %s: %w", c.BaseURL, err)
	}
	if resp.StatusCode/100 != 2 {
		defer resp.Body.Close()
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, errorBodyLimit))
		return nil, &HTTPError{Status: resp.StatusCode, Message: errorMessage(raw)}
	}
	return resp, nil
}

// errorMessage pulls the message out of the OpenAI envelope or FastAPI's
// {"detail": ...}; falls back to the raw body.
func errorMessage(raw []byte) string {
	var env struct {
		Error  *struct{ Message string } `json:"error"`
		Detail any                       `json:"detail"`
	}
	if json.Unmarshal(raw, &env) == nil {
		if env.Error != nil && env.Error.Message != "" {
			return env.Error.Message
		}
		if s, ok := env.Detail.(string); ok && s != "" {
			return s
		}
	}
	msg := strings.TrimSpace(string(raw))
	if msg == "" {
		return "no body"
	}
	return msg
}
