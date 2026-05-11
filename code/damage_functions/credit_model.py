"""
Logistic PD model for credit risk translation.

Replaces the Damodaran ICR step-function with a continuous logistic model
that produces smooth ΔPD responses to climate stress.

Based on:
  - Altman & Sabato (2007) — SME default prediction
  - Ohlson (1980) — logistic bankruptcy model
  - Shumway (2001) — hazard model with time-varying covariates

The model uses 5 financial ratios:
  1. EBIT / Total Assets      (profitability — higher → lower PD)
  2. Total Debt / Total Assets (leverage — higher → higher PD)
  3. Working Capital / TA      (liquidity — higher → lower PD)
  4. ln(Total Assets)          (size — larger → lower PD)
  5. Retained Earnings / TA    (cumulative profitability — higher → lower PD)

Climate stress enters by modifying EBIT, assets, and debt:
  - CapEx loss reduces Total Assets and Retained Earnings
  - OpEx loss reduces EBIT
  - Repair financing increases Total Debt
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class FirmFinancials:
    """Financial statement data for PD computation.

    All values in consistent units (e.g., $M).
    """
    ebit: float                 # Earnings Before Interest and Taxes
    total_assets: float         # Total Assets
    total_debt: float           # Total Debt (short + long term)
    interest_expense: float     # Annual Interest Expense
    working_capital: float      # Current Assets - Current Liabilities
    retained_earnings: float    # Retained Earnings (cumulative)
    revenue: float = 0.0       # Revenue (optional, for ICR fallback)

    @property
    def roa(self) -> float:
        """EBIT / Total Assets (profitability)."""
        return self.ebit / max(self.total_assets, 1e-6)

    @property
    def leverage(self) -> float:
        """Total Debt / Total Assets."""
        return self.total_debt / max(self.total_assets, 1e-6)

    @property
    def liquidity(self) -> float:
        """Working Capital / Total Assets."""
        return self.working_capital / max(self.total_assets, 1e-6)

    @property
    def log_size(self) -> float:
        """ln(Total Assets) — size proxy."""
        return np.log(max(self.total_assets, 1e-6))

    @property
    def cum_profit(self) -> float:
        """Retained Earnings / Total Assets."""
        return self.retained_earnings / max(self.total_assets, 1e-6)

    @property
    def icr(self) -> float:
        """Interest Coverage Ratio."""
        if self.interest_expense <= 0:
            return 99.0
        return self.ebit / self.interest_expense

    def stressed(
        self,
        capex_loss: float = 0.0,
        opex_loss: float = 0.0,
        debt_financing_ratio: float = 0.5,
        debt_interest_rate: float = 0.05,
    ) -> "FirmFinancials":
        """Create stressed financials after climate event.

        Args:
            capex_loss: direct asset damage ($M, net of insurance)
            opex_loss: indirect operational losses ($M)
            debt_financing_ratio: fraction of CapEx repair financed by new debt
            debt_interest_rate: interest rate on new debt

        Returns:
            New FirmFinancials with climate-adjusted values
        """
        new_debt = capex_loss * debt_financing_ratio
        new_interest = new_debt * debt_interest_rate

        return FirmFinancials(
            ebit=max(self.ebit - opex_loss, 0.01),
            total_assets=max(self.total_assets - capex_loss, 1.0),
            total_debt=self.total_debt + new_debt,
            interest_expense=self.interest_expense + new_interest,
            working_capital=max(self.working_capital - capex_loss * 0.3, 0),
            retained_earnings=self.retained_earnings - capex_loss - opex_loss,
            revenue=self.revenue,
        )


@dataclass
class LogisticPDParams:
    """Coefficients for logistic PD model.

    Default values calibrated to approximate Altman-Sabato (2007)
    and Ohlson (1980) for US corporate defaults.

    The model: PD = 1 / (1 + exp(-(intercept + β·x)))

    Signs convention:
      β < 0 for ratios where higher = safer (ROA, liquidity, size, cum_profit)
      β > 0 for ratios where higher = riskier (leverage)
    """
    intercept: float = -3.0     # base log-odds of default (calibrated to BBB ~ 15-30 bps)
    beta_roa: float = -8.0      # EBIT/TA: higher profitability → lower PD
    beta_leverage: float = 3.5  # Debt/TA: higher leverage → higher PD
    beta_liquidity: float = -2.0  # WC/TA: higher liquidity → lower PD
    beta_size: float = -0.25    # ln(TA): larger firms → lower PD
    beta_cum_profit: float = -3.0  # RE/TA: more retained → lower PD


class LogisticPD:
    """Continuous logistic PD model for climate credit risk.

    Usage:
        >>> firm = FirmFinancials(ebit=50, total_assets=500, total_debt=200,
        ...     interest_expense=15, working_capital=80, retained_earnings=120)
        >>> model = LogisticPD()
        >>> model.pd(firm)          # baseline PD
        >>> model.delta_pd(firm, capex_loss=30, opex_loss=20)  # climate ΔPD
        >>> model.pd_curve(firm, loss_range=(0, 100))  # PD vs loss curve
    """

    def __init__(self, params: Optional[LogisticPDParams] = None):
        self.params = params or LogisticPDParams()

    def _logit(self, firm: FirmFinancials) -> float:
        """Compute log-odds of default."""
        p = self.params
        return (
            p.intercept
            + p.beta_roa * firm.roa
            + p.beta_leverage * firm.leverage
            + p.beta_liquidity * firm.liquidity
            + p.beta_size * firm.log_size
            + p.beta_cum_profit * firm.cum_profit
        )

    def pd(self, firm: FirmFinancials) -> float:
        """Compute probability of default (continuous, 0 to 1)."""
        logit = self._logit(firm)
        logit = np.clip(logit, -20, 20)  # prevent overflow
        return float(1.0 / (1.0 + np.exp(-logit)))

    def pd_bps(self, firm: FirmFinancials) -> float:
        """PD in basis points."""
        return self.pd(firm) * 10000

    def delta_pd(
        self,
        firm: FirmFinancials,
        capex_loss: float = 0.0,
        opex_loss: float = 0.0,
        debt_financing_ratio: float = 0.5,
    ) -> dict:
        """Compute ΔPD from climate event.

        Returns dict with baseline, stressed, delta PD and all intermediates.
        """
        pd_base = self.pd(firm)
        firm_stressed = firm.stressed(capex_loss, opex_loss, debt_financing_ratio)
        pd_stressed = self.pd(firm_stressed)
        delta = pd_stressed - pd_base

        return {
            "pd_base": pd_base,
            "pd_stressed": pd_stressed,
            "delta_pd": delta,
            "delta_pd_bps": delta * 10000,
            "pd_base_bps": pd_base * 10000,
            "pd_stressed_bps": pd_stressed * 10000,
            "icr_base": firm.icr,
            "icr_stressed": firm_stressed.icr,
            "roa_base": firm.roa,
            "roa_stressed": firm_stressed.roa,
            "leverage_base": firm.leverage,
            "leverage_stressed": firm_stressed.leverage,
        }

    def pd_curve(
        self,
        firm: FirmFinancials,
        loss_range: tuple = (0, 100),
        n_points: int = 100,
        loss_split: float = 0.4,  # fraction of loss as CapEx (rest = OpEx)
    ) -> dict:
        """Generate PD vs total climate loss curve.

        Args:
            firm: baseline financials
            loss_range: (min_loss, max_loss) in $M
            n_points: number of points
            loss_split: fraction of total loss allocated to CapEx

        Returns:
            dict with loss array, pd array, delta_pd array
        """
        losses = np.linspace(loss_range[0], loss_range[1], n_points)
        pds = np.zeros(n_points)
        pd_base = self.pd(firm)

        for i, total_loss in enumerate(losses):
            capex = total_loss * loss_split
            opex = total_loss * (1 - loss_split)
            stressed = firm.stressed(capex, opex)
            pds[i] = self.pd(stressed)

        return {
            "loss": losses,
            "pd": pds,
            "delta_pd": pds - pd_base,
            "delta_pd_bps": (pds - pd_base) * 10000,
            "pd_base": pd_base,
        }

    def sensitivity(self, firm: FirmFinancials) -> dict:
        """Show which financial ratio contributes most to PD."""
        p = self.params
        contributions = {
            "intercept": p.intercept,
            "profitability (EBIT/TA)": p.beta_roa * firm.roa,
            "leverage (Debt/TA)": p.beta_leverage * firm.leverage,
            "liquidity (WC/TA)": p.beta_liquidity * firm.liquidity,
            "size (ln(TA))": p.beta_size * firm.log_size,
            "cum. profitability (RE/TA)": p.beta_cum_profit * firm.cum_profit,
        }
        total_logit = sum(contributions.values())
        return {
            "contributions": contributions,
            "total_logit": total_logit,
            "pd": float(1.0 / (1.0 + np.exp(-np.clip(total_logit, -20, 20)))),
        }
