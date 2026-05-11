"""
Sector adjustment multiplier κ: D_sector = D_residential × κ

The FEMA NFIP damage function is calibrated on residential claims.
Commercial, industrial, and infrastructure assets have different
vulnerability profiles:
  - Reinforced construction (concrete vs wood)
  - Equipment elevation (above floor level)
  - Operational resilience (redundancy, rerouting)
  - Higher insurance coverage

κ is calibrated from event study:
  κ = actual_loss / predicted_loss_residential

Calibration source: Harvey event study on 3 public companies.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class SectorAdjustment:
    """Sector-specific damage multiplier.

    κ = actual_reported_loss / model_predicted_loss (residential basis)

    Calibrated from:
      - HCA Healthcare: hospitals → κ ≈ 0.15
      - NRG Energy: power plants → κ ≈ 0.02
      - Sysco: distribution/logistics → κ ≈ 0.02

    Literature cross-check:
      - Merz et al. (2010): commercial depth-damage ≈ 40-60% of residential
      - USACE: non-residential damage factors are 50-70% of residential
      - JRC Huizinga (2017): commercial/industrial ≈ 65-85% of residential

    Our κ values are LOWER than literature because:
      1. Large companies have operational resilience (small firms don't)
      2. Insurance recovery is higher for commercial
      3. Event study companies are Fortune 500 (not representative SMEs)

    For SMEs, κ should be closer to literature values (0.4-0.7).
    """

    # calibrated from event study (large diversified companies)
    KAPPA_LARGE = {
        "healthcare":     0.15,   # HCA: hospitals, medical equipment
        "energy":         0.02,   # NRG: power plants, reinforced infrastructure
        "distribution":   0.02,   # Sysco: warehouses, logistics
        "manufacturing":  0.05,   # extrapolated: industrial equipment
        "hospitality":    0.20,   # hotels: close to residential
        "retail":         0.15,   # stores: moderate vulnerability
        "office":         0.10,   # office buildings: elevated IT
        "infrastructure": 0.03,   # bridges, roads: engineered
    }

    # literature-based (SME / generic commercial)
    KAPPA_SME = {
        "healthcare":     0.50,
        "energy":         0.30,
        "distribution":   0.40,
        "manufacturing":  0.35,
        "hospitality":    0.70,
        "retail":         0.60,
        "office":         0.40,
        "infrastructure": 0.20,
        "residential":    1.00,
    }

    # default: blend of large and SME based on company size
    REVENUE_THRESHOLD_B = 5.0  # above $5B revenue → use KAPPA_LARGE

    @classmethod
    def get_kappa(cls, sector: str, revenue_B: float = 0.0,
                  company_size: str = "auto") -> float:
        """Get sector adjustment multiplier.

        Args:
            sector: business sector (lowercase)
            revenue_B: annual revenue in $B (for auto-sizing)
            company_size: "large", "sme", or "auto"

        Returns:
            κ ∈ (0, 1] — multiply residential DR by this
        """
        sector = sector.lower()

        if company_size == "auto":
            company_size = "large" if revenue_B >= cls.REVENUE_THRESHOLD_B else "sme"

        if company_size == "large":
            kappa_dict = cls.KAPPA_LARGE
        else:
            kappa_dict = cls.KAPPA_SME

        # find best match
        for key in kappa_dict:
            if key in sector:
                return kappa_dict[key]

        # default: use SME retail as conservative fallback
        return 0.50 if company_size == "sme" else 0.10

    @classmethod
    def adjust_damage(cls, dr_residential: float, sector: str,
                      revenue_B: float = 0.0) -> float:
        """Apply sector adjustment to residential damage ratio.

        D_sector = D_residential × κ
        """
        kappa = cls.get_kappa(sector, revenue_B)
        return dr_residential * kappa
