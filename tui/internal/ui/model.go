package ui

import (
	"context"
	"time"

	"charm.land/bubbles/v2/textinput"
	"charm.land/bubbles/v2/viewport"
	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"

	"github.com/kidus-der/acceptrate/tui/internal/api"
)

const (
	// DefaultFPS is the animation frame rate; the P6 gate is >= 30.
	DefaultFPS = 30
	// eventBuffer decouples the stream goroutines from the update loop.
	eventBuffer = 64
	promptText  = "> "
	placeholder = "ask the model something…"
)

// Config wires a Model. A nil Client means no network: tests and the
// frame benchmark drive the model with messages directly.
type Config struct {
	Client        *api.Client
	URL           string
	Styles        Styles
	IsDark        bool
	FPS           int
	ReducedMotion bool
	Baseline      float64 // plain-decode tok/s from bench; 0 = unknown
	StatsOnly     bool
}

// Model is the Bubble Tea v2 model for `acceptrate chat`.
type Model struct {
	cfg   Config
	state State
	anim  Animator

	vp    viewport.Model
	input textinput.Model

	width, height int
	animating     bool

	events chan tea.Msg
	ctx    context.Context
	cancel context.CancelFunc
}

// New builds an unsized model; the first WindowSizeMsg lays it out.
func New(cfg Config) Model {
	if cfg.FPS <= 0 {
		cfg.FPS = DefaultFPS
	}
	input := textinput.New()
	input.Prompt = promptText
	input.Placeholder = placeholder
	styles := textinput.DefaultStyles(cfg.IsDark)
	styles.Focused.Prompt = cfg.Styles.Prompt
	styles.Focused.Placeholder = cfg.Styles.Muted
	input.SetStyles(styles)
	input.Focus()

	vp := viewport.New()
	vp.SoftWrap = true
	vp.MouseWheelEnabled = false

	ctx, cancel := context.WithCancel(context.Background())
	return Model{
		cfg:    cfg,
		state:  State{Baseline: cfg.Baseline},
		anim:   NewAnimator(cfg.FPS, cfg.ReducedMotion),
		vp:     vp,
		input:  input,
		events: make(chan tea.Msg, eventBuffer),
		ctx:    ctx,
		cancel: cancel,
	}
}

// Init starts the cursor blink and, with a client, the stats subscription.
func (m Model) Init() tea.Cmd {
	input := m.input
	cmds := []tea.Cmd{input.Focus()}
	if m.cfg.Client != nil {
		go runStats(m.ctx, m.cfg.Client, m.events)
		cmds = append(cmds, m.listen())
	}
	return tea.Batch(cmds...)
}

// Update routes messages; every stream event consumed re-arms listen so
// exactly one receive is outstanding at a time.
func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		return m.resize(msg.Width, msg.Height), nil
	case tea.KeyPressMsg:
		return m.handleKey(msg)
	case StatsMsg:
		return m.onStats(msg.Stats)
	case StatsErrMsg:
		m.state.Connected = false
		m.state.Err = msg.Err.Error()
		return m, m.listen()
	case ChunkMsg:
		return m.onChunk(msg.Event)
	case ChatErrMsg:
		return m.onChunk(api.ChatEvent{Err: msg.Err.Error()})
	case TickMsg:
		return m.onTick()
	}
	var cmd tea.Cmd
	m.input, cmd = m.input.Update(msg)
	return m, cmd
}

func (m Model) onStats(stats api.Stats) (tea.Model, tea.Cmd) {
	m.state = ApplyStats(m.state, stats)
	cmds := []tea.Cmd{m.listen()}
	if !m.animating && m.state.KShown != float64(m.state.KTarget) {
		m.animating = true
		cmds = append(cmds, m.tick())
	}
	return m, tea.Batch(cmds...)
}

func (m Model) onChunk(ev api.ChatEvent) (tea.Model, tea.Cmd) {
	m.state = ApplyChunk(m.state, ev)
	m = m.refreshTranscript()
	return m, m.listen()
}

func (m Model) onTick() (tea.Model, tea.Cmd) {
	if !m.animating {
		return m, nil
	}
	m.state, m.animating = StepSpring(m.state, m.anim)
	if !m.animating {
		return m, nil
	}
	return m, m.tick()
}

func (m Model) handleKey(key tea.KeyPressMsg) (tea.Model, tea.Cmd) {
	switch {
	case key.Mod == tea.ModCtrl && key.Code == 'c', key.Code == tea.KeyEscape:
		m.cancel()
		return m, tea.Quit
	case key.Code == tea.KeyEnter:
		return m.submit()
	case key.Code == tea.KeyPgUp, key.Code == tea.KeyPgDown, key.Code == tea.KeyUp, key.Code == tea.KeyDown:
		var cmd tea.Cmd
		m.vp, cmd = m.vp.Update(key)
		return m, cmd
	}
	var cmd tea.Cmd
	m.input, cmd = m.input.Update(key)
	return m, cmd
}

func (m Model) submit() (tea.Model, tea.Cmd) {
	if m.cfg.StatsOnly {
		return m, nil
	}
	next, ok := Submit(m.state, m.input.Value())
	if !ok {
		return m, nil
	}
	m.state = next
	m.input.Reset()
	m = m.refreshTranscript()
	if m.cfg.Client != nil {
		go runChat(m.ctx, m.cfg.Client, next.Transcript, m.events)
	}
	return m, nil
}

func (m Model) resize(width, height int) Model {
	m.width, m.height = width, height
	m.vp.SetWidth(width)
	m.vp.SetHeight(max(1, height-headerLines-m.inputLines()))
	m.input.SetWidth(max(1, width-lipgloss.Width(promptText)))
	return m.refreshTranscript()
}

func (m Model) inputLines() int {
	if m.cfg.StatsOnly {
		return 0
	}
	return 1
}

// refreshTranscript re-renders the viewport content and follows the tail.
func (m Model) refreshTranscript() Model {
	m.vp.SetContent(renderTranscript(m.state.Transcript, m.state.Pending, m.state.Streaming, m.width, m.cfg.Styles))
	m.vp.GotoBottom()
	return m
}

// listen receives the next stream event; nil without a client.
func (m Model) listen() tea.Cmd {
	if m.cfg.Client == nil {
		return nil
	}
	ch := m.events
	return func() tea.Msg { return <-ch }
}

func (m Model) tick() tea.Cmd {
	return tea.Tick(time.Second/time.Duration(m.cfg.FPS), func(t time.Time) tea.Msg { return TickMsg(t) })
}
