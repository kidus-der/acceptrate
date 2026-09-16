package api

import "testing"

func TestParseChatChunk(t *testing.T) {
	tests := []struct {
		name string
		data string
		want ChatEvent
		err  bool
	}{
		{
			name: "role chunk with empty content",
			data: `{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}`,
			want: ChatEvent{},
		},
		{
			name: "content delta",
			data: `{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}`,
			want: ChatEvent{Text: "Hello"},
		},
		{
			name: "finish chunk",
			data: `{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"length"}]}`,
			want: ChatEvent{Finish: "length"},
		},
		{
			name: "done sentinel",
			data: "[DONE]",
			want: ChatEvent{Done: true},
		},
		{
			name: "server error envelope becomes an error event",
			data: `{"error":{"message":"generation failed; see server log","type":"server_error","code":500}}`,
			want: ChatEvent{Err: "generation failed; see server log"},
		},
		{
			name: "malformed json is an error",
			data: `{"choices":[`,
			err:  true,
		},
		{
			name: "no choices is an error",
			data: `{"id":"x","choices":[]}`,
			err:  true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ParseChatChunk(tt.data)
			if tt.err {
				if err == nil {
					t.Fatalf("want error, got %+v", got)
				}
				return
			}
			if err != nil {
				t.Fatalf("ParseChatChunk: %v", err)
			}
			if got != tt.want {
				t.Errorf("got %+v, want %+v", got, tt.want)
			}
		})
	}
}
