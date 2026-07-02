# Why an interpretable 2x2 classifier, not an HMM

The regime classifier is a deterministic rule (YoY growth/inflation vs. trailing
5yr mean, sign gives Up/Down and Rising/Falling) rather than a latent-state model
like a Hidden Markov Model. With ~50 years of monthly data covering maybe 6-8 full
macro cycles, a latent-state model has enough free parameters (transition
probabilities, emission distributions per state) to fit noise and report it as a
"regime" — and that overfit is not falsifiable by inspection, since the states are
unobserved by construction. A rule anyone can recompute on a napkin from two public
FRED series is auditable: every label in the panel can be traced back to a specific
YoY value crossing a specific trailing mean, and a reviewer can spot-check it in
under a minute. Given the whole point of this project is that the *conditioning*
on macro regime is the credible part, the classifier itself should be the least
exotic piece, not the most.
