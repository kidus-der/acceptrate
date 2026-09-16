package ui

import (
	"math"

	"github.com/charmbracelet/harmonica"

	"github.com/kidus-der/acceptrate/tui/theme"
)

// Settle thresholds: below these the spring is treated as at rest so the
// animation tick stops and the loop goes idle.
const (
	settlePos = 0.005
	settleVel = 0.05
)

// ReducedMotionEnv is the user preference the tokens ask us to honour.
const ReducedMotionEnv = "ACCEPTRATE_REDUCED_MOTION"

// SpringMapping converts tokens.json's (stiffness k, damping c) — the
// Framer-Motion / react-spring form with unit mass — into Harmonica's
// (angular frequency ω, damping ratio ζ):
//
//	ω = sqrt(k / m)          m = 1  →  sqrt(170) ≈ 13.04 rad/s
//	ζ = c / (2 sqrt(k m))           →  26 / 26.08 ≈ 0.997
//
// so the tokens describe a spring a hair under critical damping: fast
// settle, a barely visible overshoot.
func SpringMapping(stiffness, damping float64) (omega, zeta float64) {
	const mass = 1.0
	omega = math.Sqrt(stiffness / mass)
	zeta = damping / (2 * math.Sqrt(stiffness*mass))
	return omega, zeta
}

// Animator advances a scalar toward a target with the token spring, or snaps
// when motion is reduced.
type Animator struct {
	spring  harmonica.Spring
	reduced bool
}

// NewAnimator builds the spring for a fixed frame rate.
func NewAnimator(fps int, reduced bool) Animator {
	omega, zeta := SpringMapping(theme.SpringStiffness, theme.SpringDamping)
	return Animator{spring: harmonica.NewSpring(harmonica.FPS(fps), omega, zeta), reduced: reduced}
}

// Step returns the next position and velocity.
func (a Animator) Step(pos, vel, target float64) (float64, float64) {
	if a.reduced {
		return target, 0
	}
	return a.spring.Update(pos, vel, target)
}

// Settled reports whether the spring is at rest on target.
func (a Animator) Settled(pos, vel, target float64) bool {
	return math.Abs(pos-target) < settlePos && math.Abs(vel) < settleVel
}

// ReducedMotion is true when the tokens say to honour the preference and
// the environment value asks for it ("1").
func ReducedMotion(tokensHonour bool, envValue string) bool {
	return tokensHonour && envValue == "1"
}
