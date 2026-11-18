import pandas as pd
import numpy as np


class FeatureEngineer:
    """
    Builds time-series features for demand forecasting.

    Key design decision: all features are computed from past values only
    (shifted by at least 1 day). This is mandatory — using same-day data
    would cause data leakage and produce a model that cannot generalise.
    """

    def __init__(self, lag_days=None, rolling_windows=None):
        self.lag_days = lag_days or [1, 7, 14, 30]
        self.rolling_windows = rolling_windows or [7, 14, 30]

    def build(self, df: pd.DataFrame,
              product_col="product_id",
              date_col="date",
              qty_col="quantity_sold") -> pd.DataFrame:

        df = df.copy().sort_values([product_col, date_col])

        # Calendar features — capture seasonality without needing external data
        df["day_of_week"] = df[date_col].dt.dayofweek
        df["month"] = df[date_col].dt.month
        df["quarter"] = df[date_col].dt.quarter
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["week_of_year"] = df[date_col].dt.isocalendar().week.astype(int)

        # Lag features — grouped by product to avoid cross-product contamination
        for lag in self.lag_days:
            df[f"lag_{lag}d"] = df.groupby(product_col)[qty_col].shift(lag)

        # Rolling statistics — capture recent trend and volatility
        for window in self.rolling_windows:
            shifted = df.groupby(product_col)[qty_col].shift(1)
            rolled = shifted.rolling(window)
            df[f"rolling_mean_{window}d"] = rolled.mean().reset_index(level=0, drop=True)
            df[f"rolling_std_{window}d"] = rolled.std().reset_index(level=0, drop=True)

        # Demand acceleration — detects whether demand is growing or shrinking
        g = df.groupby(product_col)[qty_col]
        df["demand_change_7d"] = (
            g.shift(1).rolling(7).mean() - g.shift(8).rolling(7).mean()
        ).reset_index(level=0, drop=True)

        # Drop rows where lag windows aren't fully populated
        warmup = max(self.lag_days + self.rolling_windows)
        df = (
            df.groupby(product_col)
            .apply(lambda x: x.iloc[warmup:])
            .reset_index(drop=True)
        )

        return df