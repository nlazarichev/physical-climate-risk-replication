"""
Compound hazard damage: multiple hazards acting simultaneously.

Wind + flood ≠ D_wind + D_flood.
A damaged roof (wind) lets water in → flood damage amplified.

Model:
  D_compound = 1 - Π(1 - Dᵢ) + ρ · Π(Dᵢ)

  where ρ is a synergy coefficient:
    ρ = 0  → independent (multiplicative survival)
    ρ > 0  → amplifying (wind weakens structure → flood does more)
    ρ < 0  → mitigating (rare, e.g. rain extinguishes wildfire)
"""

from __future__ import annotations

import numpy as np
from typing import Sequence
from .richards import RichardsDamage


class CompoundDamage:
    """Combine multiple hazard-specific damage functions.

    Usage:
        >>> flood_df = RichardsDamage(flood_params)
        >>> wind_df = RichardsDamage(wind_params)
        >>> compound = CompoundDamage([flood_df, wind_df], synergy=0.3)
        >>> compound(intensities=[2.5, 45.0])  # flood 2.5m + wind 45 m/s
    """

    def __init__(
        self,
        functions: Sequence[RichardsDamage],
        synergy: float = 0.0,
        labels: Sequence[str] | None = None,
    ):
        self.functions = list(functions)
        self.synergy = synergy
        self.labels = labels or [f"hazard_{i}" for i in range(len(functions))]

    def __call__(
        self,
        intensities: Sequence[float],
        durations: Sequence[float | None] | None = None,
    ) -> dict:
        """Compute compound damage.

        Args:
            intensities: one intensity per hazard function
            durations:   optional duration (hours) per hazard

        Returns:
            dict with individual damages, compound damage, and synergy contribution
        """
        if len(intensities) != len(self.functions):
            raise ValueError(
                f"Expected {len(self.functions)} intensities, got {len(intensities)}"
            )

        if durations is None:
            durations = [None] * len(self.functions)

        # individual damages
        individual = {}
        D_values = []
        for func, xi, tau, label in zip(
            self.functions, intensities, durations, self.labels
        ):
            d = float(func(xi, tau))
            individual[label] = d
            D_values.append(d)

        D_arr = np.array(D_values)

        # compound: 1 - Π(1 - Dᵢ) + ρ · Π(Dᵢ)
        survival_product = np.prod(1.0 - D_arr)
        independent_compound = 1.0 - survival_product
        synergy_term = self.synergy * np.prod(D_arr)
        compound = np.clip(independent_compound + synergy_term, 0.0, 1.0)

        return {
            "individual": individual,
            "independent_compound": float(independent_compound),
            "synergy_contribution": float(synergy_term),
            "compound_damage": float(compound),
        }

    def sensitivity(
        self,
        base_intensities: Sequence[float],
        hazard_index: int = 0,
        intensity_range: tuple | None = None,
        n_points: int = 100,
    ) -> dict:
        """Vary one hazard while fixing others — show compound sensitivity.

        Returns dict with intensity array and corresponding compound damages.
        """
        func = self.functions[hazard_index]
        p = func.params
        if intensity_range is None:
            intensity_range = (p.F_min, p.F_s * 1.2)

        xi_sweep = np.linspace(intensity_range[0], intensity_range[1], n_points)
        compound_damages = []
        individual_damages = []

        for xi in xi_sweep:
            intensities = list(base_intensities)
            intensities[hazard_index] = xi
            result = self(intensities)
            compound_damages.append(result["compound_damage"])
            individual_damages.append(result["individual"][self.labels[hazard_index]])

        return {
            "intensity": xi_sweep,
            "individual": np.array(individual_damages),
            "compound": np.array(compound_damages),
            "label": self.labels[hazard_index],
        }
