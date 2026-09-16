// Command acceptrate-tui is `acceptrate chat`: a Bubble Tea v2 client that
// talks HTTP to `acceptrate serve` and never loads a model itself.
//
//	go run ./cmd/acceptrate-tui --mock            # demo with no model
//	go run ./cmd/acceptrate-tui --url http://127.0.0.1:8321
//	go run ./cmd/acceptrate-tui --benchmark-frames 300
package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"time"

	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
	"github.com/charmbracelet/colorprofile"

	"github.com/kidus-der/acceptrate/tui/internal/api"
	"github.com/kidus-der/acceptrate/tui/internal/mock"
	"github.com/kidus-der/acceptrate/tui/internal/ui"
	"github.com/kidus-der/acceptrate/tui/theme"
)

const (
	// maxFPS is Bubble Tea's renderer cap.
	maxFPS = 120
	// mockStepDelay paces the demo so the spring and the strip are watchable.
	mockStepDelay = 120 * time.Millisecond
	// benchSize is the terminal the benchmark pretends to have.
	benchWidth  = 120
	benchHeight = 40
)

type options struct {
	url             string
	fps             int
	noColor         bool
	statsOnly       bool
	benchmarkFrames int
	mock            bool
	baseline        float64
}

func parseFlags(args []string) (options, error) {
	var o options
	fs := flag.NewFlagSet("acceptrate-tui", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	fs.StringVar(&o.url, "url", api.DefaultURL, "acceptrate serve base URL")
	fs.IntVar(&o.fps, "fps", ui.DefaultFPS, "animation frame rate (1-120)")
	fs.BoolVar(&o.noColor, "no-color", false, "disable colour (NO_COLOR is also honoured)")
	fs.BoolVar(&o.statsOnly, "stats-only", false, "show the instrument panel without the chat")
	fs.IntVar(&o.benchmarkFrames, "benchmark-frames", 0, "render N synthetic frames, print fps, exit")
	fs.BoolVar(&o.mock, "mock", false, "start an in-process mock server (no model) and connect to it")
	fs.Float64Var(&o.baseline, "baseline", 0, "plain-decode tok/s from bench, for the ghost marker and speedup")
	if err := fs.Parse(args); err != nil {
		return options{}, err
	}
	switch {
	case o.fps < 1 || o.fps > maxFPS:
		return options{}, fmt.Errorf("--fps must be 1..%d", maxFPS)
	case o.benchmarkFrames < 0:
		return options{}, errors.New("--benchmark-frames must be >= 0")
	case o.baseline < 0:
		return options{}, errors.New("--baseline must be >= 0")
	}
	return o, nil
}

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "acceptrate-tui:", err)
		os.Exit(1)
	}
}

func run(args []string) error {
	o, err := parseFlags(args)
	if err != nil {
		return err
	}
	if o.benchmarkFrames > 0 {
		return benchmark(o)
	}
	url := o.url
	if o.mock {
		srv := mock.New(mock.Demo(), mock.Options{StepDelay: mockStepDelay})
		defer srv.Close()
		url = srv.URL
	}
	isDark := lipgloss.HasDarkBackground(os.Stdin, os.Stdout)
	model := ui.New(ui.Config{
		Client:        api.New(url),
		URL:           url,
		Styles:        ui.NewStyles(theme.ForBackground(isDark)),
		IsDark:        isDark,
		FPS:           o.fps,
		ReducedMotion: ui.ReducedMotion(theme.ReducedMotion, os.Getenv(ui.ReducedMotionEnv)),
		Baseline:      o.baseline,
		StatsOnly:     o.statsOnly,
	})
	program := tea.NewProgram(model, programOptions(o)...)
	_, err = program.Run()
	return err
}

func programOptions(o options) []tea.ProgramOption {
	opts := []tea.ProgramOption{tea.WithFPS(o.fps)}
	if o.noColor {
		opts = append(opts, tea.WithColorProfile(colorprofile.ASCII))
	}
	return opts
}

func benchmark(o options) error {
	res := ui.Benchmark(ui.BenchConfig{
		Frames: o.benchmarkFrames, FPS: o.fps, Width: benchWidth, Height: benchHeight,
		Styles: ui.NewStyles(theme.Dark()),
	}, io.Discard)
	_, err := fmt.Printf("%d frames in %s: %.0f fps (model+view, %dx%d)\n",
		res.Frames, res.Elapsed.Round(time.Millisecond), res.FPS, benchWidth, benchHeight)
	return err
}
