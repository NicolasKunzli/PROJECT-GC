# PROJECT-GC — Codebase & Methodology Notes

This file documents design decisions, methodological choices, and implementation details
for use in future reports and by collaborators (or AI assistants) picking up the work.

---

## Project overview

Analysis of urban traffic dynamics using a percolation framework applied to simulated
loop-detector data from the Simbarca dataset (~1 570 road segments, 101 simulation
sessions, 3-minute timesteps from 08:03 to 09:57).

---

## Module structure

```
config.py            — shared constants, paths, DataLoader singleton (DL)
DataLoad.py          — DataLoader: loads all session pkl files, exposes speed arrays
processing/speed.py  — pure-numpy speed utilities (NaN handling, normalisation, PCA)
network/             — road network drawing and simplification helpers
clustering/          — k-means and Ward clustering on speed profiles
percolation/
    core.py          — percolation sweep, q_c detection, bottleneck identification
    maps.py          — spatial visualisation of percolation state
    graph_search.py  — BFS-based grid clustering (strict 4-connectivity)
gif.py               — animated GIF generation
main.py              — entry point
```

---

## Percolation methodology (Li et al. 2015)

### Reference

Li, D., Fu, B., Wang, Y., Lu, G., Berezin, Y., Stanley, H. E., & Havlin, S. (2015).
*Percolation transition in dynamical traffic network with evolving critical bottlenecks.*
PNAS, 112(3), 669–672. https://doi.org/10.1073/pnas.1419185112

### Core idea

For each road segment *ij* and time *t*, define a normalised speed ratio:

```
r_ij(t) = v_ij(t) / v_max_ij
```

At a given threshold *q*, a segment is **functional** if `r_ij ≥ q` and
**dysfunctional** (congested) otherwise. As *q* decreases from 1 → 0, functional
segments merge into larger strongly-connected components (SCCs). The **critical
threshold q_c** is the value of *q* at which the second-largest SCC is maximised —
the percolation transition point where the giant component begins to fragment.

**Bottleneck segments** are those whose speed ratio sits in the band
`[q_c − δ, q_c + δ]` (default δ = 0.01): they are the last links bridging
different functional clusters and their removal disintegrates the giant component.

### Speed normalisation — free-flow reference

**Key methodological decision:** the denominator `v_max_ij` must reflect true
free-flow speed, not the best speed observed during rush hour.

#### Why the naïve approach is wrong

An earlier version used the **95th percentile of `pred_speed` (= vdist/vtime)
within the 08:03–09:57 analysis window** as `v_max`. This is biased because the
entire window is morning rush hour. The 95th percentile of a congested window is
not free-flow — it gave `v_max ≈ 31 km/h` (network median), yielding
`q_c ≈ 0.525` and an absolute threshold of only **16.5 km/h**. That means the
model thought the network was fragmenting when roads were barely crawling, which
is physically unrealistic.

#### Rigorous approach: per-segment maximum of `ld_speed`

**Implementation** (`percolation/core.py :: compute_vmax_freeflow()`):

1. Load `ld_speed` (loop-detector speed, in m/s) from **all 101 session pkl files**,
   across **all timestamps** (07:48–end, i.e. pre-rush + rush + post-rush).
2. Concatenate into a single array of shape `(total_T, N)`.
3. Take the **per-segment absolute maximum** → shape `(N,)`.

**Why this is rigorous:**
- During low-demand periods (07:48–08:00) drivers travel near the speed limit.
- The Simbarca simulation enforces a **50 km/h speed limit**; the observed global
  maximum of `ld_speed` converges to exactly **13.889 m/s = 50.0 km/h**.
- Using the absolute maximum is equivalent to Li et al.'s "95th percentile of a
  full day including overnight free-flow" — but cleaner for a simulation where the
  speed cap is known and reached.
- The per-segment structure preserves spatial heterogeneity (different roads have
  slightly different effective free-flow speeds even under the same speed limit).

**Data snapshot (network-level summary):**

| Statistic | v_max (m/s) | v_max (km/h) |
|---|---|---|
| Segment median | 13.78 | **49.6 km/h** |
| Global max | 13.89 | **50.0 km/h** |
| Segments with v_max = NaN | 41 / 1 570 | (all-NaN across entire dataset) |

#### Speed used for the current-time numerator

`DL._ld_speed_3min` (shape `S × T × N`, loaded by DataLoader, not deleted) —
the loop-detector speed for each session during the 08:03–09:57 analysis window.
Using the same sensor (`ld_speed`) for both numerator and denominator ensures
unit consistency and avoids cross-sensor bias.

### Results under the rigorous normalisation

**Critical threshold q_c:**

| Aggregation | q_c | Absolute threshold |
|---|---|---|
| Time-mean over all sessions | **0.643** | **32.2 km/h** (64.3% of 50 km/h) |
| Session median (across 101 sessions) | **0.673** | — |
| Session range | 0.543 – 0.729 | — |

**q_c evolution during morning rush (session-averaged, per timestep):**

| Time | q_c | Threshold |
|---|---|---|
| 08:03 (congestion building) | 0.734 | 36.7 km/h |
| ~08:30–09:30 (peak rush) | ~0.643–0.653 | ~32 km/h |
| 09:57 (dissipating) | 0.618 | 30.9 km/h |

This temporal pattern — q_c highest at the onset of congestion, slightly lower at
peak — is consistent with Li et al.'s observation that q_c drops as the city enters
full rush-hour congestion.

**Physical interpretation:**
> The simulated network reaches its percolation transition when the majority of roads
> are operating at approximately **64% of their free-flow speed (~32 km/h)**. Below
> this threshold the giant functional cluster fragments into isolated sub-networks.
> The bottleneck links identified at q_c are the specific roads whose speed ratio
> sits closest to this threshold and whose congestion causes the fragmentation.

### NaN / no-data policy for percolation

Segments are excluded from the functional network (rendered gray in maps) if they
have `> 30%` missing observations in the analysis window. This uses the same
tiered policy as the clustering module (`processing/speed.py :: fill_speed_nans`):
- **Case A** (≤ 30% NaN): fill with segment temporal median, include in analysis.
- **Case C** (> 30% NaN): leave as NaN, flag `nodata_mask = True`, draw in gray.

---

## Clustering methodology

Speed profiles (shape `T × N`) are clustered spatially using k-means on
PCA-reduced features and/or Ward hierarchical clustering with network connectivity
constraint. See `clustering/` for details.

---

## Data

- **Source:** Simbarca simulation dataset
- **Sensor:** Loop detectors (`ld_speed`) + vehicle trajectory aggregates (`pred_vdist`, `pred_vtime`)
- **Coverage:** 101 simulation sessions, each spanning 07:48–10:27 (variable end)
- **Analysis window:** 08:03–09:57 (morning rush, 39 × 3-min timesteps)
- **Network:** 1 570 road segments, directed graph, UTM coordinates (X ≈ 428–433 km, Y ≈ 4 580–4 584 km)
- **Speed limit:** 50 km/h (simulation-enforced)
