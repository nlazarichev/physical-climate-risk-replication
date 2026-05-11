"""
Permafrost degradation damage function.

Unlike acute hazards (flood, wind), permafrost is a CHRONIC hazard:
  - No single "event" — gradual warming over years/decades
  - Cumulative damage: each year without remediation increases vulnerability
  - Threshold effect: when bearing capacity < load → structural failure
  - Path-dependent: damage depends on history, not just current state

Model:
  1. MAGT (mean annual ground temperature) drives ALT (active layer thickness)
  2. ALT determines pile bearing capacity via empirical relationship
  3. When capacity drops below structural load → damage begins
  4. Cumulative fatigue factor degrades capacity further each year

Key parameters:
  - MAGT: mean annual ground temperature at 3.2m depth (°C, negative = frozen)
  - ALT: active layer thickness (meters)
  - k_bearing: pile bearing capacity coefficient (1.0 = design load)
  - fatigue_rate: annual degradation from freeze-thaw cycles

References:
  - Streletskiy et al. (2019) — permafrost infrastructure risk
  - Hjort et al. (2018) — pan-Arctic infrastructure threat
  - Original Overleaf Chapter 4 — coefficient of pile strength [1.0, 1.4]
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class PermafrostParams:
    """Parameters for permafrost damage model.

    Climate:
        magt_baseline:  baseline MAGT (°C, typically -5 to -1)
        warming_rate:   annual warming (°C/year, typically 0.03-0.08)

    Geotechnical:
        alt_0:          baseline active layer thickness (m)
        alt_sensitivity: ALT increase per °C of MAGT warming (m/°C)
        pile_depth:     foundation pile depth (m)

    Structural:
        k_design:       design bearing capacity coefficient (typically 1.0)
        k_safety:       safety margin in original design (typically 1.2-1.4, from SNiP)
        fatigue_rate:   annual capacity loss from freeze-thaw cycles (%/year)

    Remediation:
        remediation_cost_frac: cost of thermosyphon/remediation as fraction of asset value
        remediation_threshold: k_bearing level triggering remediation
    """

    # climate
    magt_baseline: float = -3.0
    warming_rate: float = 0.05  # °C/year

    # geotechnical
    alt_0: float = 1.5       # meters
    alt_sensitivity: float = 0.3  # m per °C warming
    pile_depth: float = 8.0  # meters

    # structural
    k_design: float = 1.0
    k_safety: float = 1.3    # SNiP safety factor
    fatigue_rate: float = 0.005  # 0.5% per year

    # remediation
    remediation_cost_frac: float = 0.15
    remediation_threshold: float = 1.1


class PermafrostDamage:
    """Time-dependent damage function for permafrost degradation.

    Unlike Richards (instantaneous), this models cumulative degradation
    over a time horizon, producing a damage trajectory D(t).

    Usage:
        >>> params = PermafrostParams(magt_baseline=-3.0, warming_rate=0.05)
        >>> pf = PermafrostDamage(params)
        >>> trajectory = pf.trajectory(years=30)
        >>> # trajectory["damage_ratio"] — array of annual damage ratios
        >>> # trajectory["k_bearing"] — bearing capacity over time
        >>> # trajectory["remediation_year"] — when intervention needed
    """

    def __init__(self, params: PermafrostParams):
        self.params = params

    def magt(self, year: int) -> float:
        """MAGT at given year (linear warming from baseline)."""
        return self.params.magt_baseline + self.params.warming_rate * year

    def alt(self, year: int) -> float:
        """Active layer thickness at given year.

        ALT increases as MAGT warms toward 0°C.
        When MAGT > 0: permafrost has fully thawed at this depth.
        """
        p = self.params
        magt = self.magt(year)
        delta_t = magt - p.magt_baseline  # warming relative to baseline
        alt = p.alt_0 + p.alt_sensitivity * max(delta_t, 0)
        return alt

    def bearing_capacity(self, year: int, cumulative_fatigue: float = 0.0) -> float:
        """Pile bearing capacity coefficient.

        k(t) = k_safety × (1 - ALT_factor) × (1 - cumulative_fatigue)

        ALT_factor: fraction of pile in thawed ground
          - When ALT < pile_depth: partial capacity loss
          - When ALT ≥ pile_depth: complete loss of frozen-ground adhesion

        Returns k_bearing (1.0 = design load, <1.0 = overloaded)
        """
        p = self.params
        alt = self.alt(year)

        # fraction of pile in thawed (unfrozen) ground
        if alt >= p.pile_depth:
            alt_factor = 1.0  # fully thawed — no frozen-ground support
        else:
            alt_factor = alt / p.pile_depth

        # capacity = safety_margin × (1 - thaw_fraction) × (1 - fatigue)
        k = p.k_safety * (1.0 - alt_factor) * (1.0 - cumulative_fatigue)
        return max(k, 0.0)

    def damage_ratio(self, k_bearing: float) -> float:
        """Map bearing capacity to damage ratio.

        k ≥ k_design: no damage (structure within design limits)
        k < k_design: damage grows as sigmoid approaching 1.0
        k ≈ 0: total structural failure

        Uses a soft threshold (not step function) because:
          - Real structures degrade gradually (cracks, settlement)
          - Multiple piles may fail at different times
          - Partial damage (tilting) precedes collapse
        """
        p = self.params
        if k_bearing >= p.k_design:
            return 0.0

        # normalized deficit: how far below design capacity
        # deficit=0 at k=k_design, deficit=1 at k=0
        deficit = 1.0 - k_bearing / p.k_design
        deficit = np.clip(deficit, 0, 1)

        # sigmoid-like mapping: slow start, accelerating, then plateau
        # D = deficit^1.5 gives convex shape (slow onset, fast collapse)
        return float(deficit ** 1.5)

    def trajectory(
        self,
        years: int = 50,
        remediation: bool = False,
    ) -> dict:
        """Compute full damage trajectory over time horizon.

        Args:
            years:        projection horizon
            remediation:  if True, apply remediation when k drops below threshold

        Returns:
            dict with arrays indexed by year:
              - magt: ground temperature trajectory
              - alt: active layer thickness
              - k_bearing: bearing capacity coefficient
              - damage_ratio: annual damage fraction
              - cumulative_damage: running total
              - remediation_events: list of years when remediation triggered
              - remediation_cost: cumulative remediation cost (fraction of asset)
        """
        p = self.params

        magts = np.zeros(years)
        alts = np.zeros(years)
        k_bearings = np.zeros(years)
        damages = np.zeros(years)
        cum_fatigue = 0.0
        remediation_events = []
        remediation_cost = 0.0

        for t in range(years):
            magts[t] = self.magt(t)
            alts[t] = self.alt(t)

            # check remediation
            if remediation and t > 0 and k_bearings[t - 1] < p.remediation_threshold:
                # remediation resets fatigue and partially restores capacity
                cum_fatigue *= 0.3  # doesn't fully reset
                remediation_events.append(t)
                remediation_cost += p.remediation_cost_frac

            k_bearings[t] = self.bearing_capacity(t, cum_fatigue)
            damages[t] = self.damage_ratio(k_bearings[t])

            # accumulate fatigue (freeze-thaw cycles)
            if magts[t] < 0:  # still frozen — cycles continue
                cum_fatigue += p.fatigue_rate
                cum_fatigue = min(cum_fatigue, 0.5)  # cap at 50%

        return {
            "years": np.arange(years),
            "magt": magts,
            "alt": alts,
            "k_bearing": k_bearings,
            "damage_ratio": damages,
            "cumulative_damage": np.cumsum(damages),
            "remediation_events": remediation_events,
            "remediation_cost": remediation_cost,
        }

    def critical_year(self) -> Optional[int]:
        """Year when damage first exceeds 0 (k drops below k_design).

        Returns None if no damage within 100-year horizon.
        """
        traj = self.trajectory(years=100, remediation=False)
        nonzero = np.where(traj["damage_ratio"] > 0.01)[0]
        if len(nonzero) == 0:
            return None
        return int(nonzero[0])

    def plot(self, years: int = 50, remediation: bool = False, axes=None):
        """Plot damage trajectory: MAGT, ALT, k_bearing, damage."""
        import matplotlib.pyplot as plt

        traj = self.trajectory(years, remediation)

        if axes is None:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
            fig.suptitle("Permafrost Degradation Trajectory", fontsize=14)

        ax1, ax2, ax3, ax4 = axes.flat

        ax1.plot(traj["years"], traj["magt"], "r-", linewidth=2)
        ax1.axhline(0, color="k", linestyle="--", alpha=0.3, label="0°C threshold")
        ax1.set_ylabel("MAGT (°C)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(traj["years"], traj["alt"], "b-", linewidth=2)
        ax2.axhline(self.params.pile_depth, color="r", linestyle="--",
                     alpha=0.5, label=f"Pile depth ({self.params.pile_depth}m)")
        ax2.set_ylabel("ALT (m)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        ax3.plot(traj["years"], traj["k_bearing"], "g-", linewidth=2)
        ax3.axhline(self.params.k_design, color="r", linestyle="--",
                     alpha=0.5, label="Design load")
        ax3.set_ylabel("Bearing Capacity (k)")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        ax4.plot(traj["years"], traj["damage_ratio"], "darkred", linewidth=2)
        ax4.fill_between(traj["years"], 0, traj["damage_ratio"], alpha=0.2, color="red")
        for yr in traj["remediation_events"]:
            ax4.axvline(yr, color="green", linestyle=":", alpha=0.7)
        ax4.set_ylabel("Damage Ratio")
        ax4.set_xlabel("Year")
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        return axes
