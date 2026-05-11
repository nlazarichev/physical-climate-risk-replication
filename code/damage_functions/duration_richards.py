"""
Duration-dependent Richards damage function.

Key insight from FEMA calibration:
  - USACE (instantaneous) → sigmoid shape
  - Harvey (2+ weeks) → concave shape (fast onset, slow saturation)

Solution: Richards parameters themselves depend on duration τ,
smoothly transitioning from sigmoid to concave.

  a(τ) = a₀ · exp(-λ_a · τ/τ₀)       inflection shifts left
  Q(τ) = Q₀ · exp(-λ_Q · τ/τ₀)       onset accelerates
  ν(τ) = ν₀ + δ_ν · ln(1 + τ/τ₀)     asymmetry grows

At τ=0: standard Richards (sigmoid, matches USACE)
At τ→∞: concave function (matches Harvey)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class DurationRichardsParams:
    """Parameters for duration-dependent Richards.

    Core Richards (at τ=0):
        B:    steepness (1/meters for flood)
        Q_0:  initial threshold asymmetry
        a_0:  initial inflection point (normalized)
        nu_0: initial derivative asymmetry

    Duration modifiers:
        lambda_a:  rate of inflection shift (higher = faster leftward shift)
        lambda_Q:  rate of onset acceleration
        delta_nu:  magnitude of asymmetry growth
        tau_0:     reference duration scale (hours)

    Normalization:
        F_min: minimum intensity for damage
        F_s:   intensity at total destruction
    """
    # core at τ=0
    B: float = 1.5
    Q_0: float = 1.0
    a_0: float = 0.45
    nu_0: float = 0.4

    # duration modifiers
    lambda_a: float = 0.5
    lambda_Q: float = 0.3
    delta_nu: float = 0.3
    tau_0: float = 48.0  # hours — 2 days as reference

    # normalization
    F_min: float = 0.0
    F_s: float = 8.0

    @property
    def vector(self) -> np.ndarray:
        """All calibratable parameters as array."""
        return np.array([
            self.B, self.Q_0, self.a_0, self.nu_0,
            self.lambda_a, self.lambda_Q, self.delta_nu,
        ])

    @classmethod
    def from_vector(cls, v: np.ndarray, tau_0: float = 48.0,
                    F_min: float = 0.0, F_s: float = 8.0) -> "DurationRichardsParams":
        return cls(
            B=v[0], Q_0=v[1], a_0=v[2], nu_0=v[3],
            lambda_a=v[4], lambda_Q=v[5], delta_nu=v[6],
            tau_0=tau_0, F_min=F_min, F_s=F_s,
        )

    PARAM_NAMES = ["B", "Q₀", "a₀", "ν₀", "λ_a", "λ_Q", "δ_ν"]


class DurationRichards:
    """Duration-dependent Richards damage function.

    Usage:
        >>> params = DurationRichardsParams()
        >>> df = DurationRichards(params)
        >>> df(depth=2.0, duration_hours=0)    # instantaneous → sigmoid
        >>> df(depth=2.0, duration_hours=336)  # 2 weeks → concave
        >>> df.curve(duration_hours=0)         # full curve at τ=0
        >>> df.curve(duration_hours=336)       # full curve at τ=2weeks
    """

    def __init__(self, params: DurationRichardsParams):
        self.params = params

    def effective_params(self, tau: float) -> tuple[float, float, float]:
        """Compute duration-adjusted Q, a, ν.

        Returns (Q_eff, a_eff, nu_eff).
        """
        p = self.params
        ratio = tau / p.tau_0

        a_eff = p.a_0 * np.exp(-p.lambda_a * ratio)
        Q_eff = max(p.Q_0 * np.exp(-p.lambda_Q * ratio), 0.001)
        nu_eff = max(p.nu_0 + p.delta_nu * np.log(1 + ratio), 0.01)

        return Q_eff, a_eff, nu_eff

    def __call__(
        self,
        intensity: float | np.ndarray,
        duration_hours: float = 0.0,
    ) -> float | np.ndarray:
        """Compute damage ratio.

        Args:
            intensity:      flood depth in meters (or other physical unit)
            duration_hours: how long the hazard persists

        Returns:
            Damage ratio D ∈ [0, 1]
        """
        p = self.params

        # normalize intensity
        xi = (np.asarray(intensity, dtype=float) - p.F_min) / (p.F_s - p.F_min)

        # get duration-adjusted parameters
        Q_eff, a_eff, nu_eff = self.effective_params(duration_hours)

        # Richards
        exponent = -p.B * (xi - a_eff)
        exponent = np.clip(exponent, -50, 50)
        denominator = (1.0 + Q_eff * np.exp(exponent)) ** (1.0 / nu_eff)
        result = 1.0 / denominator

        return np.clip(result, 0.0, 1.0)

    def curve(
        self,
        duration_hours: float = 0.0,
        x_range: tuple = (0, 8),
        n_points: int = 200,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate full damage curve at given duration.

        Returns (intensity_array, damage_array).
        """
        xi = np.linspace(x_range[0], x_range[1], n_points)
        return xi, self(xi, duration_hours)

    def plot_duration_family(self, durations_h: list | None = None, ax=None):
        """Plot family of curves at different durations."""
        import matplotlib.pyplot as plt

        if durations_h is None:
            durations_h = [0, 6, 24, 72, 168, 336, 672]

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        cmap = plt.cm.YlOrRd
        for i, tau in enumerate(durations_h):
            color = cmap(0.2 + 0.7 * i / len(durations_h))
            xi, D = self.curve(tau)
            label = f"{tau}h" if tau < 24 else f"{tau//24}d"
            ax.plot(xi, D, color=color, linewidth=2, label=label)

        ax.set_xlabel("Flood Depth (m)")
        ax.set_ylabel("Damage Ratio")
        ax.set_title("Duration-Dependent Richards: Sigmoid → Concave Transition")
        ax.legend(title="Duration", fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        return ax


def predict_from_vector(theta: np.ndarray, xi: np.ndarray,
                        tau: float, tau_0: float = 48.0,
                        F_min: float = 0.0, F_s: float = 8.0) -> np.ndarray:
    """Predict DR from parameter vector — for calibration.

    theta = [B, Q_0, a_0, nu_0, lambda_a, lambda_Q, delta_nu]
    """
    params = DurationRichardsParams.from_vector(theta, tau_0, F_min, F_s)
    model = DurationRichards(params)
    return model(xi, tau)
