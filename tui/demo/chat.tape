# VHS tape for the `acceptrate chat` money shot: prose flows into a code
# block, alpha climbs, K springs 4 -> 6, throughput pulls away from the
# ghosted baseline. Runs against the in-process mock: no model is loaded.
#
#   vhs tui/demo/chat.tape        (from the repo root; VHS is not installed
#                                  here — https://github.com/charmbracelet/vhs)
#
# Capture face per design/tokens.json: JetBrains Mono.

Output tui/demo/chat.gif

Set Shell "zsh"
Set FontFamily "JetBrains Mono"
Set FontSize 15
Set Width 1100
Set Height 720
Set Theme "Catppuccin Mocha"
Set Padding 16
Set Framerate 30
Set PlaybackSpeed 1.0

Hide
Type "cd tui && go build -o /tmp/acceptrate-tui ./cmd/acceptrate-tui && clear"
Enter
Sleep 4s
Show

Type "/tmp/acceptrate-tui --mock --baseline 16.6"
Enter
Sleep 1.5s

Type "Why does speculative decoding get faster inside a code block?"
Sleep 800ms
Enter

# ~40 windows at 120 ms each; hold on the settled K = 6 frame.
Sleep 7s

Ctrl+C
Sleep 500ms
