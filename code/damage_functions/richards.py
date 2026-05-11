"""
Extended Richards Damage Function.

Improvements over the original Overleaf formulation:
  A. Duration effect — same intensity at different durations → different damage
  B. Dynamic OPEX ratio — nonlinear, intensity-dependent (not fixed coefficient)
  C. Uncertainty quantification — Beta-distributed output with confidence intervals
  D. Universality-aware normalization — built-in F_min/F_s scaling
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Parameter container
# ---------------------------------------------------------------------------
@dataclass
class RichardsParams:
    """Parameters of the extended Richards damage function.

    Core (original 4):
        B:   steepness — how fast damage grows (hazard-specific)
        Q:   threshold asymmetry — vulnerability of construction type
        a:   inflection point — center of the transition zone
        nu:  derivative asymmetry — where the main growth happens (left vs right)

    Extension (new):
        gamma:    duration amplification strength
        tau_0:    duration reference scale (hours)

    Normalization (universality hypothesis):
        F_min:  hazard intensity below which damage ≈ 0
        F_s:    hazard intensity at which damage ≈ 1 (total destruction)

    OPEX (dynamic):
        opex_r_base:   baseline OPEX/CAPEX ratio at low damage
        opex_r_scale:  additional OPEX multiplier at high damage
        opex_alpha:    nonlinearity exponent for OPEX growth
        opex_q:        monthly geometric decay factor (0.6–0.8)
        opex_N:        recovery horizon in months (12–24)
    """

    # --- core Richards ---
    B: float = 5.0
    Q: float = 1.0
    a: float = 0.5
    nu: float = 1.0

    # --- duration extension ---
    gamma: float = 0.3
    tau_0: float = 24.0  # hours

    # --- universality normalization ---
    F_min: float = 0.0
    F_s: float = 1.0

    # --- OPEX dynamics ---
    opex_r_base: float = 0.5
    opex_r_scale: float = 3.0
    opex_alpha: float = 1.5
    opex_q: float = 0.7
    opex_N: int = 18  # months

    @property
    def core(self) -> np.ndarray:
        """Return core parameters as array (for optimization)."""
        return np.array([self.B, self.Q, self.a, self.nu])

    @classmethod
    def from_array(cls, arr: np.ndarray, **kwargs) -> "RichardsParams":
        """Construct from optimization array [B, Q, a, nu] + optional kwargs."""
        return cls(B=arr[0], Q=arr[1], a=arr[2], nu=arr[3], **kwargs)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class RichardsDamage:
    """Extended Richards damage function with duration, OPEX dynamics,
    and uncertainty quantification.

    Usage:
        >>> params = RichardsParams(B=6, Q=0.5, a=0.5, nu=0.3, F_min=0.3, F_s=8.0)
        >>> df = RichardsDamage(params)
        >>> df(intensity=2.0)                          # point estimate
        >>> df(intensity=2.0, duration_hours=48)       # with duration
        >>> df.with_uncertainty(intensity=2.0, sigma=0.05)  # Beta CI
        >>> df.total_loss(intensity=2.0, asset_value=1e6, insurance_coverage=0.6)
    """

    def __init__(self, params: RichardsParams):
        self.params = params

    # ------------------------------------------------------------------
    # Core damage function
    # ------------------------------------------------------------------
    def __call__(
        self,
        intensity: float | np.ndarray,
        duration_hours: Optional[float] = None,
    ) -> float | np.ndarray:
        """Compute damage ratio D ∈ [0, 1].

        Args:
            intensity:      hazard intensity in physical units
                            (flood depth m, wind speed m/s, hail diameter mm)
            duration_hours: exposure duration (None = instantaneous/default)

        Returns:
            Damage ratio (fraction of asset value destroyed)
        """
        p = self.params

        # --- Step 1: normalize intensity via universality hypothesis ---
        xi = self._normalize(intensity)

        # --- Step 2: apply duration amplification ---
        if duration_hours is not None:
            xi = xi * self._duration_factor(duration_hours)

        # --- Step 3: Richards function ---
        return self._richards(xi)

    def _normalize(self, intensity: float | np.ndarray) -> float | np.ndarray:
        """Map physical intensity to [0, ~1] via F_min / F_s."""
        p = self.params
        span = p.F_s - p.F_min
        if span <= 0:
            raise ValueError(f"F_s ({p.F_s}) must be > F_min ({p.F_min})")
        return (np.asarray(intensity, dtype=float) - p.F_min) / span

    def _duration_factor(self, tau: float) -> float:
        """Amplification from prolonged exposure.

        g(τ) = 1 + γ · ln(1 + τ/τ₀)

        At τ=0: g=1 (no amplification).
        At τ=τ₀: g ≈ 1 + 0.69γ.
        Logarithmic saturation prevents runaway at very long durations.
        """
        p = self.params
        return 1.0 + p.gamma * np.log(1.0 + tau / p.tau_0)

    def _richards(self, xi: float | np.ndarray) -> float | np.ndarray:
        """Core Richards/generalized logistic.

        D(ξ) = 1 / (1 + Q·exp[-B·(ξ - a)])^(1/ν)
        """
        p = self.params
        exponent = -p.B * (xi - p.a)
        # clip to prevent overflow
        exponent = np.clip(exponent, -50, 50)
        denominator = (1.0 + p.Q * np.exp(exponent)) ** (1.0 / p.nu)
        result = 1.0 / denominator
        return np.clip(result, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Uncertainty quantification
    # ------------------------------------------------------------------
    def with_uncertainty(
        self,
        intensity: float | np.ndarray,
        sigma: float = 0.05,
        duration_hours: Optional[float] = None,
        ci: float = 0.9,
    ) -> dict:
        """Return damage ratio with Beta-distributed uncertainty.

        The mean damage D_mean is computed from Richards.
        We fit a Beta(α, β) distribution with mean=D_mean and
        prescribed standard deviation σ.

        Args:
            intensity:      hazard intensity
            sigma:          uncertainty width (std dev of damage ratio)
            duration_hours: optional duration
            ci:             confidence interval width (default 90%)

        Returns:
            dict with keys: mean, std, ci_low, ci_high, alpha, beta
        """
        from scipy.stats import beta as beta_dist

        D_mean = float(self(intensity, duration_hours))

        # clamp mean away from 0/1 to keep Beta well-defined
        D_mean = np.clip(D_mean, 0.001, 0.999)
        sigma = min(sigma, np.sqrt(D_mean * (1 - D_mean)) * 0.99)

        # Beta method-of-moments: match mean and variance
        var = sigma ** 2
        common = D_mean * (1 - D_mean) / var - 1
        alpha = D_mean * common
        beta_param = (1 - D_mean) * common

        tail = (1 - ci) / 2
        ci_low = float(beta_dist.ppf(tail, alpha, beta_param))
        ci_high = float(beta_dist.ppf(1 - tail, alpha, beta_param))

        return {
            "mean": D_mean,
            "std": sigma,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "alpha": alpha,
            "beta": beta_param,
        }

    # ------------------------------------------------------------------
    # OPEX dynamics
    # ------------------------------------------------------------------
    def opex_ratio(self, damage_ratio: float) -> float:
        """Dynamic OPEX/CAPEX ratio — grows nonlinearly with damage.

        ratio(D) = r_base + r_scale · D^α

        At D≈0: ratio ≈ r_base  (minor disruption)
        At D≈1: ratio ≈ r_base + r_scale  (total shutdown, supply chain collapse)

        Replaces fixed industry coefficients from original model.
        """
        p = self.params
        return p.opex_r_base + p.opex_r_scale * (damage_ratio ** p.opex_alpha)

    def opex_monthly(self, damage_ratio: float, asset_value: float) -> np.ndarray:
        """Monthly OPEX loss schedule with geometric decay.

        OPEX_k = total_opex · q^k · (1-q) / (1-q^(N+1))

        Returns array of length N with monthly OPEX losses.
        """
        p = self.params
        capex_loss = damage_ratio * asset_value
        total_opex = capex_loss * self.opex_ratio(damage_ratio)

        k = np.arange(p.opex_N)
        weights = p.opex_q ** k
        weights *= (1 - p.opex_q) / (1 - p.opex_q ** (p.opex_N + 1))

        return total_opex * weights

    # ------------------------------------------------------------------
    # Full loss calculation
    # ------------------------------------------------------------------
    def total_loss(
        self,
        intensity: float,
        asset_value: float,
        insurance_coverage: float = 0.0,
        duration_hours: Optional[float] = None,
    ) -> dict:
        """Complete loss estimate: CAPEX + OPEX with insurance offset.

        Args:
            intensity:          hazard intensity
            asset_value:        replacement value of the asset
            insurance_coverage: fraction covered by insurance [0, 1]
            duration_hours:     optional duration

        Returns:
            dict with capex_loss, opex_total, opex_monthly, net_loss, gross_loss
        """
        D = float(self(intensity, duration_hours))

        capex_gross = D * asset_value
        capex_net = capex_gross * (1 - insurance_coverage)

        opex_monthly = self.opex_monthly(D, asset_value)
        opex_total = float(opex_monthly.sum())

        return {
            "damage_ratio": D,
            "capex_gross": capex_gross,
            "capex_net": capex_net,
            "opex_total": opex_total,
            "opex_monthly": opex_monthly,
            "gross_loss": capex_gross + opex_total,
            "net_loss": capex_net + opex_total,
        }

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    def plot(
        self,
        intensity_range: Optional[tuple] = None,
        duration_hours: Optional[float] = None,
        sigma: float = 0.05,
        ax=None,
        label: Optional[str] = None,
    ):
        """Plot damage curve with uncertainty band."""
        import matplotlib.pyplot as plt

        p = self.params
        if intensity_range is None:
            intensity_range = (p.F_min, p.F_s * 1.2)

        xi = np.linspace(intensity_range[0], intensity_range[1], 300)
        D = self(xi, duration_hours)

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))

        ax.plot(xi, D, linewidth=2, label=label or "D(ξ)")

        # uncertainty band
        ci_low = np.array([
            self.with_uncertainty(x, sigma, duration_hours)["ci_low"] for x in xi
        ])
        ci_high = np.array([
            self.with_uncertainty(x, sigma, duration_hours)["ci_high"] for x in xi
        ])
        ax.fill_between(xi, ci_low, ci_high, alpha=0.2, label="90% CI")

        ax.set_xlabel("Hazard Intensity (ξ)")
        ax.set_ylabel("Damage Ratio D(ξ)")
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax
