package api

import (
	"encoding/json"
	"errors"
	"fmt"
)

// doneSentinel ends an OpenAI stream (serve/sse.py DONE_LINE).
const doneSentinel = "[DONE]"

// ChatEvent is one decoded chat stream frame. Exactly one of Text, Finish,
// Done or Err is meaningful per frame; the role chunk decodes to a zero value.
type ChatEvent struct {
	Text   string
	Finish string
	Done   bool
	Err    string
}

// Message is one turn of the conversation sent to /v1/chat/completions.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatChunk struct {
	Choices []struct {
		Delta struct {
			Content string `json:"content"`
		} `json:"delta"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error"`
}

// ParseChatChunk decodes the data of one chat stream frame.
func ParseChatChunk(data string) (ChatEvent, error) {
	if data == doneSentinel {
		return ChatEvent{Done: true}, nil
	}
	var chunk chatChunk
	if err := json.Unmarshal([]byte(data), &chunk); err != nil {
		return ChatEvent{}, fmt.Errorf("chat chunk: %w", err)
	}
	if chunk.Error != nil {
		return ChatEvent{Err: chunk.Error.Message}, nil
	}
	if len(chunk.Choices) == 0 {
		return ChatEvent{}, errors.New("chat chunk: no choices")
	}
	choice := chunk.Choices[0]
	return ChatEvent{Text: choice.Delta.Content, Finish: choice.FinishReason}, nil
}
