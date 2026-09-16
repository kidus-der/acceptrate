package main

import "testing"

func TestParseFlags(t *testing.T) {
	tests := []struct {
		name string
		args []string
		want options
		err  bool
	}{
		{
			name: "defaults",
			args: nil,
			want: options{url: "http://127.0.0.1:8321", fps: 30},
		},
		{
			name: "everything set",
			args: []string{"--url", "http://localhost:9000", "--fps", "60", "--no-color",
				"--stats-only", "--benchmark-frames", "300", "--mock", "--baseline", "16.6"},
			want: options{url: "http://localhost:9000", fps: 60, noColor: true, statsOnly: true,
				benchmarkFrames: 300, mock: true, baseline: 16.6},
		},
		{"fps must be positive", []string{"--fps", "0"}, options{}, true},
		{"fps above the renderer cap", []string{"--fps", "121"}, options{}, true},
		{"negative benchmark frames", []string{"--benchmark-frames", "-1"}, options{}, true},
		{"negative baseline", []string{"--baseline", "-2"}, options{}, true},
		{"unknown flag", []string{"--bogus"}, options{}, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parseFlags(tt.args)
			if tt.err {
				if err == nil {
					t.Fatalf("want error, got %+v", got)
				}
				return
			}
			if err != nil {
				t.Fatalf("parseFlags: %v", err)
			}
			if got != tt.want {
				t.Errorf("got %+v, want %+v", got, tt.want)
			}
		})
	}
}
