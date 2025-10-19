# 🌧️ Cloudburst Prediction Using Machine Learning
A practicum project with Alt Surya Inc. that involves building ML models that predict cloudbursts in a 3h time span

**Columbia University · QMSS Data Analysis Practicum (Spring 2025)**  
*Collaborators: Yuxi Xiao, Hanyi Xu, Kendrick Yu, Kevin Chien, Jackie Cheng, Ariel Shao, Linda Xi*  

---

## 🚀 Project Overview
Cloudbursts are sudden, highly localized rainfall events exceeding **100 mm/hour**, often causing flash floods, infrastructure damage, and severe socioeconomic losses.  
This project develops a **machine-learning-based early-warning system** to predict cloudburst occurrences within a **3-hour window**, using NOAA’s Local Climatological Data (LCD).

---

## 🎯 Objectives
- Detect **cloudburst events** within the next **3 hours**.  
- Achieve accuracy **above the 50% baseline** from prior literature.  
- Improve early-warning and disaster-preparedness frameworks through reproducible, interpretable ML pipelines.

---

## 📚 Literature Foundation
| Study | Methodology | Key Insight |
|--------|--------------|-------------|
| Murakami et al. (2022) | Autoencoder for anomaly detection | Identifies precipitation anomalies but lacks event-specific recall |
| Patil & Kulkarni (2023) | GFS + XGBoost/Neural Network | High short-term forecasting accuracy (F1 ≈ 0.84) |
| Shani & Nagappan (2024) | CNN with Gramian Angular Fields | Captures spatial–temporal rainfall patterns (Acc ≈ 86%) |

Our model integrates **supervised learning** with **feature-engineered meteorological inputs** from NOAA, bridging insights from these approaches.

---

## 🧩 Data Source
**NOAA Local Climatological Data (LCD)**  
- Station ID: `72202012839` (Miami International Airport)  
- Period: **2015 – 2024**  
- Frequency: **Hourly**  
- Variables: temperature (dry/wet bulb, dew point), pressure, humidity, wind, precipitation  
- Records: ~100,000 observations  

---

## ⚙️ Data Cleaning & Feature Engineering
1. **Preprocessing**
   - Selected relevant hourly variables and standardized timestamps.
   - Prioritized reports by format type (FM-15 > FM-12 > FM-16 > SOD > SOM).  
   - Removed duplicates/outliers; applied linear interpolation for moderate missingness.

2. **Feature Engineering**
   - **Temporal features:** hour, month, weekend flag, sine/cosine time cycles.  
   - **Derived meteorological variables:** vapor pressure, U/V wind components, precipitation intensity.  
   - **Lag features:** 1-h, 3-h, 6-h rolling windows to capture atmospheric transitions.  
   - **Interaction terms:** temperature × humidity, pressure × wind, etc.  
   - **Physics-based indicators:**  
     - Clausius-Clapeyron–based moisture capacity.  
     - Rapid temperature change flags preceding storms.  
   - **Imbalance handling:** applied **SMOTE** to oversample rare cloudburst cases.

---

## 🧠 Modeling Approach
| Model | Description | Strengths |
|--------|--------------|------------|
| **Random Forest** | 100 trees, depth = 10, min_samples_split = 5 | Robust to noise, interpretable feature importance |
| **Neural Network (Experimental)** | Dense architecture, ReLU activation | Tests deep-learning potential for short-term forecasting |

**Training Strategy:**  
- Temporal train/test split to preserve chronological order.  
- Evaluated using **Precision, Recall, F1-score, and Confusion Matrices.**

---

## 📈 Key Results
| Model Variant | Precision | Recall | F1-Score |
|----------------|------------|---------|-----------|
| Random Forest (baseline) | 0.50 | 0.005 | 0.0099 |
| + Interaction Terms | 0.24 | 0.008 | 0.016 |
| + Dry Period Duration | **1.00** | **0.030** | **0.060** |

> ⚠️ Models achieved high precision but very low recall — excellent at avoiding false alarms but often missing real events.  
> Future work aims to rebalance this trade-off via **cost-sensitive learning**, **SMOTE-ENN**, and **hybrid anomaly–classification frameworks**.

---

## 🧮 Example Feature Engineering Snippet
```python
# Example: 3-hour rolling changes
df['temp_change_3h'] = df['DryBulbTemp'].diff(3)
df['precip_sum_6h'] = df['HourlyPrecipitation'].rolling(6).sum()
df['pressure_change'] = df['StationPressure'].diff(1)

# Moisture capacity via Clausius–Clapeyron
df['moisture_capacity'] = 6.11 * np.exp(17.62 * df['DewPoint'] / (243.12 + df['DewPoint']))
```

---

## 💡 Evaluation Insights
- **Top predictors:** Hourly precipitation, pressure change, wind speed, dry-bulb temperature, relative humidity.  
- **Challenge:** Severe class imbalance; model identified most non-events but missed many true cloudbursts.  
- **Interpretation:** The conservative prediction threshold maximized precision but limited recall — critical in early-warning contexts.

---

## 🔐 Ethical & Practical Considerations
- **Precision vs. Recall trade-off:** Overly conservative models risk missing real disasters.  
- **Transparency:** Emphasis on interpretable models and clear uncertainty communication.  
- **Data Integrity:** NOAA data is public and anonymized; integrity and reproducibility prioritized.

---

## 🧭 Future Directions
- Implement **cost-sensitive** and **ensemble hybrid** models.  
- Integrate **multi-geographic datasets** (e.g., Southeast Asia, Hawaii).  
- Enhance explainability via **SHAP feature attributions**.  
- Explore **real-time dashboards** and cloud-based deployment (Streamlit/AWS).

---

## 🤖 AI Tools Reflection
Throughout development, **ChatGPT** and **Claude** assisted with:
- Code debugging, syntax clarification, and algorithm selection.  
- Structuring literature reviews and refining technical writing.  
All outputs were independently verified against actual data and metrics.

---

## 🧾 Repository Structure
```
├── data/
│   └── NOAA_LCData.csv
├── notebooks/
│   ├── data_preprocessing.ipynb
│   ├── feature_engineering.ipynb
│   ├── model_training_randomforest.ipynb
│   └── evaluation_visualization.ipynb
├── results/
│   ├── confusion_matrix
│   ├── feature_importance
│   └── model_metrics
├── README.md
```

---
