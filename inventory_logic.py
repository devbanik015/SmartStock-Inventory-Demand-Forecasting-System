import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.stats import norm


@dataclass
class InventoryParams:
    lead_time_days: int
    service_level: float        # e.g. 0.95
    holding_cost_per_unit: float
    ordering_cost: float


class InventoryAnalyser:
    """
    Translates demand forecasts into actionable inventory decisions.

    Implements standard operations research formulas:
    - Safety stock:   Z × σ × √lead_time
    - Reorder point:  (avg_daily_demand × lead_time) + safety_stock
    - EOQ:            √(2DS / H)
    """

    def __init__(self, params: InventoryParams):
        self.p = params
        self.z = norm.ppf(params.service_level)

    def safety_stock(self, demand_std: float) -> float:
        return self.z * demand_std * np.sqrt(self.p.lead_time_days)

    def reorder_point(self, avg_daily_demand: float, demand_std: float) -> float:
        return avg_daily_demand * self.p.lead_time_days + self.safety_stock(demand_std)

    def eoq(self, annual_demand: float) -> float:
        return np.sqrt(
            (2 * annual_demand * self.p.ordering_cost) / self.p.holding_cost_per_unit
        )

    def classify_risk(self, current_stock: float,
                       rop: float,
                       avg_daily_demand: float) -> str:
        if avg_daily_demand <= 0:
            return "NO_DEMAND"
        days_left = current_stock / avg_daily_demand
        if current_stock <= 0:
            return "OUT_OF_STOCK"
        if current_stock < rop:
            return "REORDER_NOW"
        if days_left < self.p.lead_time_days * 1.5:
            return "LOW_STOCK"
        return "HEALTHY"

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        df must have: product_id, avg_daily_demand, demand_std,
                      current_stock, annual_demand
        """
        rows = []
        for _, r in df.iterrows():
            ss = self.safety_stock(r["demand_std"])
            rop = self.reorder_point(r["avg_daily_demand"], r["demand_std"])
            q = self.eoq(r["annual_demand"])
            risk = self.classify_risk(r["current_stock"], rop, r["avg_daily_demand"])
            days_left = (
                round(r["current_stock"] / r["avg_daily_demand"], 1)
                if r["avg_daily_demand"] > 0 else None
            )
            rows.append({
                "product_id": r["product_id"],
                "safety_stock": round(ss, 1),
                "reorder_point": round(rop, 1),
                "eoq": round(q, 1),
                "days_of_stock": days_left,
                "risk_status": risk,
            })
        return pd.DataFrame(rows)