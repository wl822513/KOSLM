# KOSLM

**Kalman-Optimal Selective Long-Term Memory**

KOSLM reformulates LSTM as a structured state-space model and embeds a Kalman-optimal selective gating mechanism. It enables context-aware long-term sequence modeling, improving forecasting accuracy, stability, and scalability. KOSLM demonstrates strong performance across standard benchmarks and real-world Secondary Surveillance Radar (SSR) target tracking.

---

## Research Overview

Long Short-Term Memory networks (LSTMs) are widely used for long-term sequence modeling but suffer from information decay and gradient instability on long sequences. KOSLM addresses these issues by:

* **State-space reformulation of LSTM**: Treats each gate as parameterizing a time-varying SSM based on input and hidden states.
* **Kalman-optimal selective mechanism**: Introduces a Kalman gain-based gating pathway that minimizes latent state estimation uncertainty, enabling context-aware information selection.
* **Theoretical and empirical validation**: Kalman gain convergence is theoretically guaranteed under linear-Gaussian assumptions and confirmed via synthetic experiments. KOSLM achieves 10–30% lower MSE than state-of-the-art baselines on long-term forecasting tasks.
* **Real-world robustness**: Demonstrated on SSR target tracking with noisy, sparse, and irregular sampling.

KOSLM maintains near-linear scalability and achieves up to **2.5× speedup** over Mamba-2 on long sequences while using only 0.24M parameters.

---

## Dataset Information

This project involves multiple long-term time series datasets. They are not included due to size. Please download and place them in the `data/` folder. Filenames should match those specified in the code.

---

### 1. Electricity

* **Dataset Name**: ElectricityLoadDiagrams20112014
* **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/electricityloaddiagrams20112014)
* **Description**: Electricity consumption data of 370 clients, sampled every 15 minutes, from 2011-01-01 to 2014-12-31.

---

### 2. ETT-small

* **Dataset Name**: ETT-small (Electricity Transformer Dataset)
* **Source**: [GitHub Repository](https://github.com/zhouhaoyi/ETDataset)
* **Description**: Transformer electricity load data, sampled every minute over 2 years, regions: ETT-small-m1 / ETT-small-m2.

---

### 3. Exchange Rate

* **Dataset Name**: Exchange Rate Time Series
* **Source**: [GitHub Project](https://github.com/bala-1409/Foreign-Exchange-Rate-Time-Series-Data-science-Project)
* **Description**: Daily exchange rates of multiple currency pairs, time span varies per pair.

---

### 4. Traffic

* **Dataset Name**: Traffic Time Series Dataset
* **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/stealthtechnologies/traffic-time-series-dataset)
* **Description**: Hourly traffic flow data for urban roads over 1 year.

---

### 5. Weather

* **Dataset Name**: Jena Climate Dataset
* **Source**: [Keras Example](https://keras.io/examples/timeseries/timeseries_weather_forecasting/)
* **Description**: Weather data in Jena, Germany, sampled every 10 minutes with 14 features including temperature, humidity, wind speed, and pressure.

---

### 6. SSR (Secondary Surveillance Radar) Case Study

We also evaluate KOSLM on a real-world SSR application: real-time trajectory prediction and tracking.

* **Dataset Name**: SSR Plots Dataset
* **Source**: [IEEE DataPort](https://ieee-dataport.org/documents/10.21227/qxmq-z688)
* **Description**: Radar plots collected from field-deployed SSR systems, used to demonstrate KOSLM’s robustness under noisy and irregular sampling conditions.
* **DOI**: 10.21227/QXMQ-Z688

---

## Usage Instructions

1. Download the datasets and place the CSV files in the `data/` folder.
2. Ensure filenames match those used in the code.
3. You may download via direct links, Kaggle API, or GitHub API as needed.

---

## Installation & Requirements

```bash
# Clone the repository
git clone https://github.com/yourusername/KOSLM.git
cd KOSLM

# Create environment
conda create -n koslm python=3.10
conda activate koslm

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

```python
from koslm import KOSLM
from data_loader import load_dataset

# Load dataset
data = load_dataset("ETT-small-m1")

# Initialize model
model = KOSLM(input_dim=data.shape[1], hidden_dim=128)

# Train
model.train(data, epochs=50)

# Predict
predictions = model.predict(data)
```


