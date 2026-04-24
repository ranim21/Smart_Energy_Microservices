# Energy Forecasting & Anomaly Detection

> End-to-end ML pipeline for real-time energy 
> consumption analysis · Sofiatech · 2025

## 🔍 Problem
Energy consumption anomalies and unpredictable 
demand patterns lead to unnecessary costs and 
operational inefficiencies.

## 💡 Solution
A complete ML pipeline combining time-series 
forecasting and autoencoder-based anomaly detection, 
fed by real-time API data streams.

## 📊 Results
| Model | R² | MAE | RMSE |
|-------|-----|-----|------|
| Bidirectional LSTM ✅ | 0.95 | 6,750 | 7,019 |
| ARIMA | 0.94 | 7,500 | 12,000 |
| Naive Forecast | -0.49 | 8,448 | 14,728 |

## 🛠️ Stack
Python · LSTM · ARIMA · Autoencoder · 
Plotly · Docker · FastAPI · Scikit-learn

## 🚀 How to run
pip install -r requirements.txt
python orchestrator.py
