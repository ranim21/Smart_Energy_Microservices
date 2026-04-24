# ⚡ Smart Energy Microservices
### AI-Powered Energy Forecasting, Anomaly Detection & Recommendations

> End-to-end ML pipeline built for IoT platform integration · Sofiatech · 2025  
> Real-time time-series data · Microservices architecture · Dockerized deployment

---

## 🔍 Problem

Energy consumption anomalies and unpredictable demand patterns lead to unnecessary costs and operational inefficiencies. Traditional monitoring systems lack predictive capabilities — they react instead of anticipate.

---

## 💡 Solution

A modular, production-ready AI pipeline with three core intelligence layers:

| Module | Goal | Approach |
|--------|------|----------|
| 🔮 **Forecasting** | Predict future energy consumption | LSTM · ARIMA · Moving Average |
| 🚨 **Anomaly Detection** | Identify unusual consumption patterns | K-Means Clustering · PCA |
| 💡 **Recommendations** | Generate actionable cost-saving insights | Rule-based AI engine |

---

## 📊 Results

### Forecasting — Model Benchmark (8 models evaluated)

| Model | MAE (Test) | RMSE (Test) | R² (Test) |
|-------|-----------|-------------|-----------|
| Naive Forecast | 8,448 | 14,728 | -0.49 |
| Moving Average (k=12) | **11.32** | **15.60** | **0.9938** ✅ |
| ARIMA(5,0,1) | 95,030 | 1,424,622 | -5.19e7 |
| AutoReg | 7,500 | 12,000 | -0.25 |
| Simple LSTM | 1,146 | 9,987 | 0.9488 |
| Stacked LSTM | 2,729 | 10,847 | 0.1917 |
| Bidirectional LSTM | 12,430 | 13,752 | -0.2993 |
| **SeqToSeq LSTM** ✅ | **6,750** | **7,019** | **0.055** |

> **Selected model** : SeqToSeq LSTM for production forecasting — best balance of MAE and generalization on unseen data.

### Anomaly Detection
- **Method** : K-Means Clustering with 2D PCA projection
- Anomalies highlighted and visualized in real-time dashboards
- Low false positive rate validated on real energy consumption data

---

## 🏗️ Architecture

```
Smart_Energy_Microservices/
├── data_ingestion_service/     # Real-time API data collection
├── preprocessing_service/      # Temporal aggregation, imputation, feature engineering
├── forecasting_service/        # LSTM & statistical model inference
├── anomaly_service/            # K-Means clustering & anomaly flagging
├── orchestrator_script.py      # Pipeline orchestration
└── config_common.py            # Shared configuration
```

### Preprocessing Pipeline
The preprocessing service handles:
- **Temporal Aggregation & Resampling** (5-minute intervals)
- **Missing Value Imputation** (Forward fill · KNN · Linear interpolation · Zero-fill)
- **Outlier Detection & Treatment** (MAD-based)
- **Feature Scaling** (MinMax Scaler)
- **Feature Engineering** (Month, day of week, weekend flags)

---

## 🛠️ Stack

```
Python · LSTM (PyTorch/Keras) · Scikit-learn · ARIMA (statsmodels)
Plotly · Docker · FastAPI · K-Means · PCA · Pandas · NumPy
```

---

## 🚀 How to Run

### Prerequisites
```bash
Python 3.9+
Docker (optional, recommended)
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the full pipeline
```bash
python orchestrator_script.py
```

### Run individual services
```bash
# Data ingestion
python data_ingestion_service/main.py

# Preprocessing
python preprocessing_service/main.py

# Forecasting
python forecasting_service/main.py

# Anomaly detection
python anomaly_service/main.py
```

### Run with Docker
```bash
docker-compose up --build
```

---

## 📈 Key Visualizations

- Real-time energy consumption dashboards (Plotly)
- K-Means anomaly detection with 2D PCA projection
- Forecasting results with confidence intervals
- Preprocessing workflow validation plots

---

## 👩‍💻 Author

**Ranim Bouzamoucha** — AI Engineer  
📍 Paris, France  
🔗 [LinkedIn](https://linkedin.com/in/ranim-bouzamoucha)  

---

## 📄 License

MIT License — feel free to use and adapt with attribution.
