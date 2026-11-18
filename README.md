# 📦 SmartStock — Inventory Demand Forecasting System

A machine learning system that forecasts product demand from historical sales data,
identifies stock risk, and delivers inventory insights through an interactive dashboard.

Built with Python, Pandas, Scikit-learn, and Streamlit.

---

## 🖼️ Dashboard Preview

> *Upload a sales CSV and get per-product forecasts, reorder alerts, and inventory KPIs in seconds.*

---

## 🧠 Motivation

Businesses lose revenue from two inventory failure modes — stockouts and overstocking.
Most small-to-medium operations manage stock reactively. SmartStock shifts that to a
data-driven approach using historical sales patterns to forecast demand and trigger
reorder decisions automatically.

---

## ⚙️ Features

- **Data ingestion** — accepts CSV or Excel POS exports, handles messy dates, duplicates, and returns
- **Feature engineering** — lag features, rolling statistics, demand acceleration, calendar signals
- **Demand forecasting** — Random Forest and Gradient Boosting with walk-forward (chronological) validation
- **Inventory logic** — safety stock, reorder point, and EOQ using operations research formulas
- **Risk classification** — flags products as `OUT_OF_STOCK`, `REORDER_NOW`, `LOW_STOCK`, or `HEALTHY`
- **Interactive dashboard** — per-product drill-down, forecast chart, feature importance, model metrics

---

## 🗂️ Project Structure

```
smartstock/
├── data/
│   ├── raw/                  # Original sales exports (not committed)
│   └── processed/            # Cleaned, aggregated data
├── src/
│   ├── __init__.py
│   ├── generate_data.py      # Synthetic data generator for testing
│   ├── preprocessor.py       # Cleaning and daily aggregation
│   ├── feature_engineering.py# Lag, rolling, and calendar features
│   ├── forecaster.py         # Scikit-learn model wrapper
│   └── inventory_logic.py    # Safety stock, EOQ, reorder point
├── app/
│   └── dashboard.py          # Streamlit dashboard
├── notebooks/
│   └── 01_eda.ipynb          # Exploratory data analysis
├── tests/
│   └── test_inventory.py     # Unit tests for inventory logic
├── requirements.txt
└── README.md
```

---

## 🚀 Quickstart

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/smartstock.git
cd smartstock
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Generate sample data**
```bash
python src/generate_data.py
```

**5. Launch the dashboard**
```bash
streamlit run app/dashboard.py
```

Then open `http://localhost:8501` and upload the generated `data/raw/sales.csv`.

---

## 📊 Model Design Decisions

### Why walk-forward validation?
A random train/test split on time-series data leaks future demand into training.
Walk-forward validation keeps the hold-out set as the most recent N days — the
only split that reflects how the model will actually be used in production.

### Why Random Forest over ARIMA?
ARIMA is a strong baseline for univariate series but requires stationarity checks
and manual parameter tuning per product. Random Forest handles multiple products
in one model, incorporates external features (calendar, promotions), and degrades
gracefully when data is noisy. The trade-off is interpretability — mitigated here
by feature importance scores.

### Known limitation
During recursive multi-step forecasting, only `lag_1d` updates at each step.
Longer lags (7d, 14d, 30d) hold their last-known values, which underestimates
uncertainty at t+7 and beyond. A proper fix would recompute all lag features
from the growing forecast window — marked as a TODO in `forecaster.py`.

---

## 📐 Inventory Formulas

| Metric | Formula |
|---|---|
| Safety Stock | `Z × σ_demand × √lead_time` |
| Reorder Point | `(avg_demand × lead_time) + safety_stock` |
| EOQ | `√(2 × D × S / H)` |

Where Z is the service-level z-score, σ is demand standard deviation,
D is annual demand, S is ordering cost, and H is holding cost per unit.

---

## 🧪 Running Tests

```bash
python -m pytest tests/ -v
```

---

## 📦 Requirements

```
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.0
streamlit==1.35.0
plotly==5.22.0
scipy==1.13.1
joblib==1.4.2
openpyxl==3.1.3
```

---

## 🔭 Future Improvements

- [ ] Recompute all lag features during recursive forecasting for better uncertainty estimates
- [ ] Add Prophet as a third model option for stronger seasonality handling
- [ ] Incorporate external signals (promotions, public holidays)
- [ ] Export reorder alerts as a CSV report from the dashboard
- [ ] Containerise with Docker for consistent deployment

---

## 👤 Author

**Your Name**
[LinkedIn](https://linkedin.com/in/yourprofile) · [GitHub](https://github.com/yourusername)
