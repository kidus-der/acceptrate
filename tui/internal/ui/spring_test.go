package ui

import (
	"math"
	"testing"

	"github.com/kidus-der/acceptrate/tui/theme"
)

func TestSpringMappingFromTokens(t *testing.T) {
	omega, zeta := SpringMapping(theme.SpringStiffness, theme.SpringDamping)

	// omega = sqrt(k/m) with m = 1; zeta = c / (2 sqrt(k m)).
	if math.Abs(omega-math.Sqrt(170)) > 1e-9 {
		t.Errorf("omega = %v, want sqrt(170)", omega)
	}
	if math.Abs(zeta-26/(2*math.Sqrt(170))) > 1e-9 {
		t.Errorf("zeta = %v", zeta)
	}
	if zeta >= 1 || zeta < 0.95 {
		t.Errorf("zeta = %.3f: tokens 170/26 should be just under critically damped", zeta)
	}
}

func TestAnimatorSettlesOnTargetWithinASecond(t *testing.T) {
	const fps = 30
	a := NewAnimator(fps, false)
	pos, vel := 4.0, 0.0

	frames := 0
	for ; frames < fps*3; frames++ {
		pos, vel = a.Step(pos, vel, 6)
		if a.Settled(pos, vel, 6) {
			break
		}
	}

	if frames >= fps {
		t.Errorf("took %d frames to settle, want < %d", frames, fps)
	}
	if frames < 3 {
		t.Errorf("settled in %d frames: that is a snap, not a spring", frames)
	}
	if math.Abs(pos-6) > 0.01 {
		t.Errorf("pos = %v, want ~6", pos)
	}
}

func TestAnimatorMovesMonotonicallyTowardTargetAtFirst(t *testing.T) {
	a := NewAnimator(30, false)
	pos, vel := 4.0, 0.0

	p1, v1 := a.Step(pos, vel, 6)
	p2, _ := a.Step(p1, v1, 6)

	if !(pos < p1 && p1 < p2 && p2 <= 6.05) {
		t.Errorf("positions %v -> %v -> %v should climb toward 6", pos, p1, p2)
	}
}

func TestReducedMotionSnaps(t *testing.T) {
	a := NewAnimator(30, true)

	pos, vel := a.Step(4, 0, 6)

	if pos != 6 || vel != 0 {
		t.Errorf("reduced motion: pos=%v vel=%v, want 6/0", pos, vel)
	}
	if !a.Settled(pos, vel, 6) {
		t.Error("should be settled immediately")
	}
}

func TestReducedMotionPreference(t *testing.T) {
	tests := []struct {
		name   string
		tokens bool
		env    string
		want   bool
	}{
		{"tokens honour, env set", true, "1", true},
		{"tokens honour, env unset", true, "", false},
		{"tokens honour, env zero", true, "0", false},
		{"tokens ignore, env set", false, "1", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := ReducedMotion(tt.tokens, tt.env); got != tt.want {
				t.Errorf("ReducedMotion(%v, %q) = %v, want %v", tt.tokens, tt.env, got, tt.want)
			}
		})
	}
}
