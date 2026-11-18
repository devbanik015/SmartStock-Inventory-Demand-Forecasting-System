import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib


class DemandForecaster:
    """
    Scikit-learn wrapper for demand forecasting.

    Uses walk-forward (chronological) validation — a random train/test
    split is incorrect for time-series because it leaks future information
    into the training set.
    """

    FEATURE_COLS = [
        "day_of_week", "month", "quarter", "is_weekend", "week_of_year",
        "lag_1d", "lag_7d", "lag_14d", "lag_30d",
        "rolling_mean_7d", "rolling_mean_14d", "rolling_mean_30d",
        "rolling_std_7d", "rolling_std_30d",
        "demand_change_7d",
    ]

    def __init__(self, model_type: str = "random_forest"):
        self.model_type = model_type
        self.model = self._build_model()
        self.is_fitted = False

    def _build_model(self):
        if self.model_type == "random_forest":
            return RandomForestRegressor(
                n_estimators=200, max_depth=10,
                min_samples_leaf=5, random_state=42, n_jobs=-1
            )
        elif self.model_type == "gradient_boosting":
            return GradientBoostingRegressor(
                n_estimators=200, learning_rate=0.05,
                max_depth=4, random_state=42
            )
        raise ValueError(f"Unknown model: {self.model_type}")

    def _get_X(self, df: pd.DataFrame) -> np.ndarray:
        cols = [c for c in self.FEATURE_COLS if c in df.columns]
        return df[cols].values

    def walk_forward_validate(self, df: pd.DataFrame,
                               target_col="quantity_sold",
                               test_days=30) -> dict:
        df = df.sort_values("date")
        cutoff = df["date"].max() - pd.Timedelta(days=test_days)

        train = df[df["date"] <= cutoff]
        test = df[df["date"] > cutoff]

        self.model.fit(self._get_X(train), train[target_col].values)
        self.is_fitted = True

        preds = np.clip(self.model.predict(self._get_X(test)), 0, None)
        y = test[target_col].values

        mask = y > 0
        mape = np.mean(np.abs((y[mask] - preds[mask]) / y[mask])) * 100

        return {
            "mae": round(mean_absolute_error(y, preds), 3),
            "rmse": round(np.sqrt(mean_squared_error(y, preds)), 3),
            "mape": round(mape, 2),
            "train_rows": len(train),
            "test_rows": len(test),
        }

    def forecast(self, df: pd.DataFrame, horizon: int = 30) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Call walk_forward_validate() before forecasting.")

        future = df.copy()
        results = []

        for _ in range(horizon):
            last = future.iloc[[-1]]
            pred = max(0.0, self.model.predict(self._get_X(last))[0])
            next_date = last["date"].values[0] + pd.Timedelta(days=1)

            new_row = last.copy()
            new_row["date"] = next_date
            new_row["quantity_sold"] = pred
            new_row["lag_1d"] = pred

            results.append({"date": next_date, "forecast": round(pred, 2)})
            future = pd.concat([future, new_row], ignore_index=True)

        return pd.DataFrame(results)

    def feature_importance(self) -> pd.DataFrame:
        if not self.is_fitted or not hasattr(self.model, "feature_importances_"):
            return pd.DataFrame()
        cols = [c for c in self.FEATURE_COLS]
        return (
            pd.DataFrame({"feature": cols, "importance": self.model.feature_importances_})
            .sort_values("importance", ascending=False)
        )

    def save(self, path: str):
        joblib.dump(self.model, path)

    def load(self, path: str):
        self.model = joblib.load(path)
        self.is_fitted = True