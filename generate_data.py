import pandas as pd
import numpy as np

def generate_sales_data(n_products: int = 10,
                        days: int = 365,
                        seed: int = 42) -> pd.DataFrame:
    """
    Generates realistic synthetic sales data with:
    - Weekly seasonality (weekends spike for some products)
    - Monthly seasonality
    - Random noise
    - Occasional stockout gaps (zero demand)
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=days, freq="D")
    records = []

    for i in range(n_products):
        base_demand = rng.integers(10, 80)
        weekly_amp = rng.uniform(0.1, 0.4)
        monthly_amp = rng.uniform(0.05, 0.2)

        for j, date in enumerate(dates):
            seasonal = (
                1
                + weekly_amp * np.sin(2 * np.pi * date.dayofweek / 7)
                + monthly_amp * np.sin(2 * np.pi * date.month / 12)
            )
            noise = rng.normal(1.0, 0.15)
            qty = max(0, int(base_demand * seasonal * noise))

            # Simulate occasional stockouts
            if rng.random() < 0.03:
                qty = 0

            records.append({
                "date": date,
                "product_id": f"SKU-{1000 + i}",
                "quantity_sold": qty,
                "unit_price": round(rng.uniform(5, 150), 2),
            })

    return pd.DataFrame(records)


if __name__ == "__main__":
    df = generate_sales_data()
    df.to_csv("data/raw/sales.csv", index=False)
    print(f"Generated {len(df)} rows across {df['product_id'].nunique()} products.")