"""
End-to-end damage assessment pipeline.

Asset → Hazard Overlay → Damage Function → Loss → Credit Translation

Covers modules 2-5 from Novikov (2026) taxonomy:
  Module 2: Exposure mapping (asset geocoding)
  Module 3: Vulnerability / damage functions (Richards)
  Module 4: Financial loss estimation (CAPEX + OPEX)
  Module 5: Credit risk translation (LogisticPD-based ΔPD)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from .richards import RichardsDamage, RichardsParams
from .credit_model import LogisticPD, FirmFinancials


@dataclass
class Asset:
    """A physical asset exposed to climate risk."""

    name: str
    lat: float
    lon: float
    value: float                       # replacement value
    asset_class: str = "commercial"    # commercial, industrial, residential
    construction: str = "concrete"     # concrete, steel, wood, mixed
    insurance_coverage: float = 0.0    # fraction insured [0, 1]


@dataclass
class HazardExposure:
    """Result of overlaying an asset with hazard maps."""

    asset: Asset
    hazard_type: str           # flood, wind, hail, permafrost
    intensity: float           # in physical units
    duration_hours: float = 0  # for floods: how long submerged
    return_period: float = 0   # e.g., 1-in-100-year event


@dataclass
class CreditMetrics:
    """Output credit risk metrics following BCBS framework."""

    # firm identifiers
    firm_name: str = ""
    ebit: float = 0.0
    interest_expense: float = 0.0
    total_assets: float = 0.0

    # climate-adjusted
    climate_loss: float = 0.0
    icr_baseline: float = 0.0
    icr_stressed: float = 0.0
    delta_pd: float = 0.0
    pd_baseline: float = 0.0
    pd_stressed: float = 0.0


class DamagePipeline:
    """Full pipeline: exposures → damage → financial loss → credit metrics.

    Usage:
        >>> pipe = DamagePipeline()
        >>> pipe.register_hazard("flood", RichardsDamage(flood_params))
        >>> pipe.register_hazard("wind", RichardsDamage(wind_params))
        >>>
        >>> exposures = [
        ...     HazardExposure(asset, "flood", intensity=1.5, duration_hours=72),
        ...     HazardExposure(asset, "wind", intensity=35),
        ... ]
        >>> loss = pipe.assess(exposures)
        >>> credit = pipe.credit_translation(loss, ebit=5e6, interest=1e6)
    """

    def __init__(self):
        self._hazard_funcs: dict[str, RichardsDamage] = {}
        self._pd_model = LogisticPD()

    def register_hazard(self, hazard_type: str, func: RichardsDamage):
        """Register a calibrated damage function for a hazard type."""
        self._hazard_funcs[hazard_type] = func

    def assess(self, exposures: list[HazardExposure]) -> dict:
        """Assess damage across all hazard exposures for one borrower.

        Returns aggregated loss dictionary.
        """
        total_capex_gross = 0.0
        total_capex_net = 0.0
        total_opex = 0.0
        details = []

        for exp in exposures:
            func = self._hazard_funcs.get(exp.hazard_type)
            if func is None:
                raise KeyError(
                    f"No damage function registered for '{exp.hazard_type}'. "
                    f"Available: {list(self._hazard_funcs.keys())}"
                )

            loss = func.total_loss(
                intensity=exp.intensity,
                asset_value=exp.asset.value,
                insurance_coverage=exp.asset.insurance_coverage,
                duration_hours=exp.duration_hours if exp.duration_hours > 0 else None,
            )

            total_capex_gross += loss["capex_gross"]
            total_capex_net += loss["capex_net"]
            total_opex += loss["opex_total"]

            details.append({
                "asset": exp.asset.name,
                "hazard": exp.hazard_type,
                "intensity": exp.intensity,
                "duration_h": exp.duration_hours,
                "damage_ratio": loss["damage_ratio"],
                "capex_gross": loss["capex_gross"],
                "capex_net": loss["capex_net"],
                "opex_total": loss["opex_total"],
            })

        return {
            "total_capex_gross": total_capex_gross,
            "total_capex_net": total_capex_net,
            "total_opex": total_opex,
            "total_gross": total_capex_gross + total_opex,
            "total_net": total_capex_net + total_opex,
            "details": details,
        }

    def credit_translation(
        self,
        loss: dict,
        ebit: float,
        interest_expense: float,
        total_assets: float = 0.0,
        firm_name: str = "",
    ) -> CreditMetrics:
        """Translate financial loss to credit risk via LogisticPD model.

        Uses the 5-ratio logistic model (Altman-Sabato / Ohlson-inspired)
        instead of the Damodaran ICR step function.

        For firms where only ebit and interest_expense are known,
        we estimate total_assets, total_debt, working_capital, and
        retained_earnings using typical manufacturing ratios.
        """
        capex_net = loss["total_capex_net"]
        opex_total = loss["total_opex"]

        # Build FirmFinancials (estimate missing values)
        ta = total_assets if total_assets > 0 else ebit * 5
        firm = FirmFinancials(
            ebit=ebit,
            total_assets=ta,
            total_debt=interest_expense / 0.05,
            interest_expense=interest_expense,
            working_capital=ta * 0.15,
            retained_earnings=ta * 0.25,
        )

        pd_result = self._pd_model.delta_pd(firm, capex_loss=capex_net, opex_loss=opex_total)

        return CreditMetrics(
            firm_name=firm_name,
            ebit=ebit,
            interest_expense=interest_expense,
            total_assets=ta,
            climate_loss=loss["total_net"],
            icr_baseline=pd_result["icr_base"],
            icr_stressed=pd_result["icr_stressed"],
            delta_pd=pd_result["delta_pd"],
            pd_baseline=pd_result["pd_base"],
            pd_stressed=pd_result["pd_stressed"],
        )
