package ui

import (
	"bytes"
	"io"
	"strings"
	"testing"
	"time"

	"github.com/kidus-der/acceptrate/tui/theme"
)

// The P6 gate: the TUI renders at >= 30 fps while generating. 300 frames
// in under 10 s is 30 fps with a wide margin on a busy machine.
func TestBenchmarkRendersThirtyFramesPerSecondWhileGenerating(t *testing.T) {
	const frames = 300

	res := Benchmark(BenchConfig{Frames: frames, FPS: 30, Width: 120, Height: 40,
		Styles: NewStyles(theme.Dark())}, io.Discard)

	if res.Frames != frames {
		t.Fatalf("rendered %d frames, want %d", res.Frames, frames)
	}
	if res.Elapsed >= 10*time.Second {
		t.Fatalf("300 frames took %v (%.1f fps), want < 10 s", res.Elapsed, res.FPS)
	}
	if res.FPS < 30 {
		t.Errorf("achieved %.1f fps, want >= 30", res.FPS)
	}
}

func TestBenchmarkFramesAreBusyGeneratingFrames(t *testing.T) {
	var out bytes.Buffer

	res := Benchmark(BenchConfig{Frames: 40, FPS: 30, Width: 100, Height: 30,
		Styles: NewStyles(theme.Dark())}, &out)

	if res.Frames != 40 || out.Len() == 0 {
		t.Fatalf("frames=%d bytes=%d", res.Frames, out.Len())
	}
	last := out.String()[strings.LastIndex(out.String(), "acceptrate chat"):]
	for _, want := range []string{"generating", "tok/s", "K = ", "▌"} {
		if !strings.Contains(last, want) {
			t.Errorf("last frame lacks %q", want)
		}
	}
}
