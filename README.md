# CO3-AGENT0 — RC Trajectory Toy

Definitions (fixed in [rc_trajectory.py](rc_trajectory.py)):
- Space: $6\times 6$ integer grid with Manhattan metric, obstacles at $(2,3)$, $(3,1)$, $(4,4)$.
- Feasible($t,s$): neighbors from {stay, east, north, west, south} that are in-bounds and not obstacles.
- Boundary: any out-of-bounds cell or obstacle; violation if a move lands there.
- Margin: $\text{Margin}(s)=\min(d_{edge}, d_{obstacle})$ using Manhattan distance; zero on violation.
- Coherence window $\Delta$: fraction of the modal action in the last $\Delta$ steps (default 3).
- Trend: $\big(\text{Coherence}_{\text{recent}}-\text{Coherence}_{\text{prev}}\big)/(\Delta\,\Delta t)$ once two windows exist; else 0.
- Select: $\arg\min_a J$, where $J=\lambda/\text{Margin}(s')+\mu\,\mathbf{1}[a\neq a_{\text{prev}}]$, $s'$ is the candidate state, Project is identity.

Run:
1) Optional: create venv if desired. No dependencies beyond Python 3.10+.
2) Execute:

```
python rc_trajectory.py
```

Outputs: tab-separated time series per step (state, action, feasible size, selected and minimum margin, coherence, trend) plus boundary_crossed flag.

Observed runs (defaults):
- Run A start (1,1): feasible size stayed 5, selected margin 1.00, trend 0 across all windows, boundary_crossed False.
- Run B start (3,2): feasible size >=4, coherence climbed to 1.00 by step 3, trend non-negative (peak 0.111) over windows, boundary_crossed False. Qualitative behavior (stabilizing to stay) matches Run A.

Falsification statement: Feasible collapses (and trajectory halts) if the current state and all neighbor candidates are boundary cells—for example, starting on an obstacle/out-of-bounds point or surrounding the state with obstacles so every move violates the boundary.