# ArlingtonParkingPredict

End-to-end pipeline for predicting hourly parking occupancy hot spots in Arlington, VA. The project focuses on downloading transaction data from the Arlington Open Data Portal, engineering occupancy features, training a LightGBM regression model, and serving outputs through a Streamlit dashboard.


## Quick Start

```bash
python3 -m venv venv
. venv/bin/activate
pip install -r requirements.txt
```

Update `config.yaml` with paths and parameters appropriate for your environment, then run:

```bash
make train   # executes scripts/train_occupancy_model.py
make run     # launches Streamlit dashboard
```