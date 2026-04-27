# 🚆 Train-Route-Analysis-and-Journey-Time-Prediction

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![Pandas](https://img.shields.io/badge/Pandas-2.0-darkblue?style=for-the-badge&logo=pandas)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-1.3-orange?style=for-the-badge&logo=scikit-learn)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.8-green?style=for-the-badge)
![Seaborn](https://img.shields.io/badge/Seaborn-0.13-teal?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge)

**A complete end-to-end Data Science project analyzing Indian Railway records to predict train journey duration using Machine Learning.**

*Data Science Internship Project — Sysslan IT Solutions*
*By **Vishvi** | Data Scientist Intern*

[📓 View Notebook](#) &nbsp;|&nbsp; [📄 View Presentation](#) &nbsp;|&nbsp; [📊 View Dashboard](#) &nbsp;|&nbsp; [🔗 LinkedIn](#)

</div>

---

## 📖 The Story Behind This Project

> *"Imagine standing at a busy Indian railway station — bags in hand, clock ticking — and the board says: Estimated Time: Unknown."*

That one frustrating, all-too-familiar moment is exactly what this project set out to solve.

India runs one of the largest railway networks in the world — **13,000+ trains daily**, connecting villages to megacities, mountains to coastlines. Yet for most passengers, schedulers, and planners, one question remains unanswered:

**"How long will this journey actually take?"**

This project takes **1,86,074 real Indian Railway stop records**, cleans them, explores them, visualises them — and ultimately builds a Machine Learning model that answers that question with **97.65% accuracy**, using just two simple inputs: distance and number of stops.

No guesswork. No uncertainty. Just data — telling the truth.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset Summary](#-dataset-summary)
- [Project Pipeline](#-project-pipeline)
- [Level 1 — Data Overview](#-level-1--data-overview)
- [Level 2 — Data Cleaning & Feature Engineering](#-level-2--data-cleaning--feature-engineering)
- [Level 3 — Exploratory Data Analysis](#-level-3--exploratory-data-analysis)
- [Level 4 — Visualization & Pattern Analysis](#-level-4--visualization--pattern-analysis)
- [Level 5 — Prediction Model Development](#-level-5--prediction-model-development)
- [Level 6 — Final Prediction System](#-level-6--final-prediction-system)
- [Key Insights](#-key-insights)
- [Model Results](#-model-results)
- [How to Run](#-how-to-run)
- [Project Structure](#-project-structure)
- [Future Improvements](#-future-improvements)
- [About](#-about)

---

## 🎯 Project Overview

| Attribute | Detail |
|---|---|
| **Project Title** | Train Route Analysis & Journey Time Prediction |
| **Organisation** | Sysslan IT Solutions |
| **Intern** | Vishvi — Data Scientist Intern |
| **Domain** | Transportation · Data Science · Machine Learning |
| **Algorithm** | Linear Regression |
| **Dataset** | Indian Railways Stop Records |
| **Total Records** | 1,86,074 rows |
| **Model Accuracy** | R² = 0.9765 (97.65%) |
| **Tools Used** | Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn |
| **Environment** | Google Colab / Jupyter Notebook |

---

## 📦 Dataset Summary

The dataset contains stop-level records from the Indian Railways network — every row represents one train at one station.

```
Total Records        →  1,86,074 rows
Unique Trains        →  11,113
Unique Stations      →  8,147
Total Columns        →  13
Null Values          →  Zero
Duplicate Rows       →  Zero
Longest Route        →  4,260 km  (Kanyakumari → Dibrugarh)
Shortest Route       →  3 km
Median Route Dist    →  38 km
Median Stops         →  9
Avg Dwell Time       →  2.3 minutes
Median Train Speed   →  54 km/h
```

**Columns in the dataset:**

| Column | Description |
|---|---|
| `Train_No` | Unique train identifier |
| `Train_Name` | Name of the train |
| `SN` | Stop sequence number |
| `Station_Code` | Station code |
| `Station_Name` | Full station name |
| `Route_Number` | Route identifier |
| `Arrival_time` | Arrival time at station (HH:MM:SS) |
| `Departure_Time` | Departure time from station (HH:MM:SS) |
| `Distance` | Cumulative distance from origin (km) |
| `1A / 2A / 3A / SL` | Class availability flags |
| `Duration_Mins` | Cumulative journey duration in minutes |

---

## 🔄 Project Pipeline

```
Raw Data (1,86,074 rows)
        │
        ▼
  L1 · Data Overview
  ── Structure · Stats · Null Check · Route Table
        │
        ▼
  L2 · Data Cleaning & Feature Engineering
  ── Time Standardisation · Duration Calculation · Train Categories
        │
        ▼
  L3 · Exploratory Data Analysis
  ── Correlations · Station Traffic · Distance vs Duration
        │
        ▼
  L4 · Visualization & Pattern Analysis
  ── 9 Charts · Heatmap · Bar Charts · Box Plots · Scatter Plots
        │
        ▼
  L5 · Prediction Model Development
  ── Linear Regression · Train/Test Split · MAE · RMSE · R²
        │
        ▼
  L6 · Final Prediction System
  ── Live Predictions · Actual vs Predicted · Residual Analysis
```

---

## 📊 Level 1 — Data Overview

> *"Before you tell a story, you need to understand the world it lives in."*

The first step was getting to know the dataset — its size, structure, and completeness.

**What we found:**

- ✅ **Zero null values** — the dataset was remarkably clean
- ✅ **Zero duplicates** — every record was unique
- 🚉 **8,147 stations** spread across the entire country
- 🚆 **11,113 trains** — from 3 km local services to 4,260 km cross-country hauls

**Train-wise Route Summary (sample):**

| Train No | Origin | Destination | Distance | Stops |
|---|---|---|---|---|
| 11001 | CSMT Mumbai | Pune Jn | 192 km | 8 |
| 12951 | Mumbai Central | New Delhi | 1,384 km | 5 |
| 16381 | Chennai | Kanyakumari | 693 km | 19 |

---

## 🧹 Level 2 — Data Cleaning & Feature Engineering

> *"Raw data is like an uncut diamond — full of potential, but needing refinement before it shines."*

**What we cleaned:**
- Converted all `HH:MM:SS` time strings → total minutes from midnight
- Flagged origin rows (`Arrival = 00:00:00`) and terminus rows (`Departure = 00:00:00`)
- Removed edge cases with zero distance or negative duration

**Features engineered:**

| Feature | Description |
|---|---|
| `Journey_Duration_Mins` | Total minutes from origin to terminus |
| `Journey_Duration_Hrs` | Same, expressed in hours |
| `Avg_Speed_kmh` | Distance ÷ Duration |
| `Distance_Bucket` | Categorical: <50km, 50–150km, 150–500km, 500–1000km, >1000km |
| `Stop_Category` | Express / Semi-Fast / Regular / Passenger |

**Train Category Breakdown:**

```
Express   (≤ 5 stops)   ████████████████████  50%
Semi-Fast (6–15 stops)  ████████████          28%
Regular   (16–30 stops) ██████                15%
Passenger (30+ stops)   ███                    7%
```

---

## 🔍 Level 3 — Exploratory Data Analysis

> *"When the numbers whisper, you listen. When they shout — you build a model."*

### Correlation Analysis

The most important discovery of the entire project:

| Feature | Correlation with Journey Time |
|---|---|
| **Distance** | **r = 0.99** ← near-perfect |
| Number of Stops | r = 0.51 |
| Average Speed | r = 0.33 |

A correlation of **0.99** is extraordinary. Distance alone explains **99% of the variance** in journey time. Before we wrote a single line of ML code — the data had already told us what to do.

### Station Traffic Findings

**Top 10 Busiest Stations:**

```
CST Mumbai      ████████████████████████████  1,027 stops
Kalyan Jn       ████████████████████████      828
Thane           ███████████████████████       796
Sealdah         ██████████████████████        745
Chennai Beach   ██████████████████████        738
Howrah Jn       █████████████████████         699
Dadar           ██████████████████            598
Dum Dum Jn      ██████████████                463
Kurla           ██████████████                462
Tambaram        █████████████                 434
```

**Insight:** 3 of the top 4 stations are in Mumbai's suburban corridor. The CST–Kalyan–Thane triangle is the beating heart of Indian Railways.

---

## 📈 Level 4 — Visualization & Pattern Analysis

### Chart 1 — Journey Duration by Distance Bucket

> *"The perfect staircase — every step clean, every step proportional."*

| Distance | Avg Duration | Visual |
|---|---|---|
| < 50 km | 52 min | ▓ |
| 50–150 km | 115 min | ▓▓ |
| 150–500 km | 410 min | ▓▓▓▓▓▓▓ |
| 500–1000 km | 894 min | ▓▓▓▓▓▓▓▓▓▓▓▓▓▓ |
| > 1000 km | 2,116 min | ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ |

The box plots confirmed wide spread in short-distance routes — reflecting the mix of suburban express and slow local trains.

---

### Chart 2 — Station-wise Train Traffic

Top 15 busiest stations vs Bottom 15 least-served stations — showing the dramatic inequality in India's railway access. Mumbai dominates the top. Remote rural stations at the bottom serve just one train a day.

---

### Chart 3 — Distance vs Journey Duration Scatter + Heatmap

```
Journey    │                              ✦
Duration   │                         ✦ ✦
(minutes)  │                    ✦ ✦✦
           │               ✦ ✦✦✦
           │          ✦✦✦✦✦
           │      ✦✦✦✦✦
           │  ✦✦✦✦
           └─────────────────────────────────
                     Distance (km)
                  y = 1.19x + 23.1
```

Points are coloured by number of stops — revealing that **more stops = more time, even at the same distance.**

**Correlation Heatmap values:**

```
              Distance  Stops  Duration  Speed
Distance  │   1.00     0.46    0.99     0.37
Stops     │   0.46     1.00    0.51     0.14
Duration  │   0.99     0.51    1.00     0.33
Speed     │   0.37     0.14    0.33     1.00
```

---

### Chart 4 — Dwell Time & Segment Speed

| Metric | Value |
|---|---|
| Most common dwell time | **1 minute** |
| Average dwell time | **2.3 minutes** |
| Median segment speed | **54 km/h** |
| Typical speed range | 37–65 km/h |

> *"Indian trains don't waste time standing still. Delays happen between stations — not at them."*

---

## 🤖 Level 5 — Prediction Model Development

### Dataset Preparation

```python
Features  →  X = ['Total_Distance_km', 'Num_Stops']
Target    →  y = 'Journey_Duration_Mins'
Split     →  80% Train / 20% Test  (random_state=42)
Training  →  1,580 trains
Testing   →  395 trains
```

### Model Training

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
```

### The Model Equation

```
Journey Duration (min) = 0.88
                       + (1.0981 × Distance_km)
                       + (2.7807 × Num_Stops)
```

**In plain English:**
- Every extra kilometre adds → **~1.1 minutes**
- Every extra stop adds → **~2.8 minutes**

### Feature Coefficients

```
Distance (km)  ██████████████████████████  1.0981 min/km
Num Stops      ████████████████████████████████████  2.7807 min/stop
```

---

## 🎯 Level 6 — Final Prediction System

### Model Performance

| Metric | Value | Interpretation |
|---|---|---|
| **R² Score** | **0.9765** | Model explains 97.65% of all variance ✅ |
| **MAE** | **33.84 min** | Average prediction error ✅ |
| **RMSE** | **66.74 min** | Root mean squared error ✅ |
| **MAPE** | **25.62%** | Mean absolute percentage error ⚠️ |
| **Mean Residual** | **1.1 min** | Near-zero bias ✅ |

### Actual vs Predicted

```
Predicted  │                              ✦ ✦
(minutes)  │                         ✦ ✦✦
           │                    ✦ ✦✦✦       ← Points hug the
           │               ✦ ✦✦✦              diagonal tightly
           │          ✦✦✦✦✦
           │      ✦✦✦✦✦
           │  ✦✦✦✦
           └─────────────────────────────────
                     Actual (minutes)
                  R² = 0.9765
```

### MAE by Distance Bucket

| Bucket | MAE | Context |
|---|---|---|
| < 50 km | **15 min** | ✅ Excellent |
| 50–150 km | **36 min** | ✅ Very Good |
| 150–500 km | **117 min** | ⚠️ Moderate |
| 500–1000 km | **136 min** | ⚠️ Acceptable |
| > 1000 km | **168 min** | ⚠️ 8% of journey — still operationally useful |

### Live Prediction Examples

| Route Type | Distance | Stops | Predicted Time |
|---|---|---|---|
| Short suburban | 32 km | 3 | **44 minutes** |
| Short regional | 78 km | 4 | **98 minutes** |
| Inter-city | 324 km | 17 | **404 minutes (6.7 hrs)** |
| Long-distance | 655 km | 23 | **784 minutes (13 hrs)** |
| Very long haul | 1,949 km | 45 | **2,266 minutes (37.8 hrs)** |

---

## 💡 Key Insights

> *"Like chapters in a book, each insight built on the last — painting a complete picture of how India moves."*

1. 🏆 **Distance is king** — r = 0.99 correlation with journey time. It explains everything.
2. 🏙️ **Mumbai dominates** — CST, Kalyan, and Thane account for 3 of India's top 4 busiest stations.
3. 🚉 **50% of trains are express** — most Indian trains are short suburban services under 50 km.
4. ⏱️ **Dwell time is negligible** — average stop time is just 2.3 minutes. Delays are between stations.
5. 📏 **Every km adds 1.1 min** — clean, linear, predictable relationship throughout the dataset.
6. 🛑 **Every stop adds 2.8 min** — number of halts is the second most powerful predictor.
7. 🎯 **Short routes are most accurate** — MAE of just 15 minutes for routes under 50 km.
8. 🌍 **India's longest route spans 4,260 km** — Kanyakumari to Dibrugarh — the ultimate train epic.

---

## 🚀 How to Run

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/vishvi31/train-route-analysis.git
cd train-route-analysis

# 2. Add the dataset to /data folder
# (Cleaned_Train_Data.csv)

# 3. Open the notebook
jupyter notebook Train_Route_Analysis.ipynb

# OR run in Google Colab
# Upload the CSV when prompted
```

### Quick Prediction

```python
import pickle

# Load saved model
with open('models/linear_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Predict journey time
def predict_journey(distance_km, num_stops):
    pred = model.predict([[distance_km, num_stops]])[0]
    print(f"📍 Distance  : {distance_km} km")
    print(f"🛑 Stops     : {num_stops}")
    print(f"⏱️  Predicted : {pred:.1f} min  ({pred/60:.2f} hrs)")
    return round(pred, 2)

# Example
predict_journey(500, 20)
# ⏱️ Predicted: 577.8 min (9.63 hrs)
```

---

## 📁 Project Structure

```
train-route-analysis/
│
├── 📁 data/
│   ├── Dataset1.csv                    ← Raw dataset
│   └── Cleaned_Train_Data.csv          ← Cleaned dataset
│
├── 📁 notebooks/
│   └── Train_Route_Analysis.ipynb      ← Full notebook (all 6 levels)
│
├── 📁 outputs/
│   ├── level4_task4_1.png              ← Journey duration chart
│   ├── level4_task4_2.png              ← Station traffic chart
│   ├── level4_task4_3.png              ← Distance vs duration + heatmap
│   └── level6_final_dashboard.png      ← Complete ML dashboard
│
├── 📁 models/
│   └── linear_regression_model.pkl     ← Saved trained model
│
├── 📁 presentation/
│   └── Train_Route_Analysis.pdf        ← Full PDF presentation
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🔮 Future Improvements

| Improvement | Expected Impact |
|---|---|
| Add train category (Rajdhani / Express / Passenger) | R² → above 0.85 |
| Include departure hour (peak vs off-peak) | Reduce MAPE significantly |
| Try Random Forest / Gradient Boosting | Better long-haul accuracy |
| Add dwell time as a feature | Improve stop-heavy route predictions |
| Deploy as Streamlit web app | Make predictions accessible to anyone |
| Build REST API with Flask | Enable real-time operational use |

---

## 📄 Resources

| Resource | Link |
|---|---|
| 📓 Jupyter Notebook | [View Notebook](#) |
| 📊 Final Dashboard | [View Dashboard](#) |
| 📄 PDF Presentation | [Download PDF](#) |
| 🎯 Gamma Presentation | [View Slides](#) |
| 🔗 LinkedIn Post | [View Post](#) |

---

## 👩‍💻 About

**Vishvi** — Data Scientist Intern at Sysslan IT Solutions

- 🎓 BA (Hons) English Literature — School of Open Learning, University of Delhi
- 📜 Diploma in Information Technology
- 📊 IBM Data Science Professional Certificate — Coursera
- 💼 Internship: Sysslan IT Solutions
- 🔗 [LinkedIn](#) | [GitHub](#)

> *"My background in English Literature taught me that the best insights — like the best stories — need to be understood by everyone, not just experts. That's the philosophy behind every analysis in this project."*

---

## 📜 License

This project is licensed under the MIT License.

---

<div align="center">

**Built with ❤️ by Vishvi | Sysslan IT Solutions Internship 2026**

*"Good data science doesn't just crunch numbers. It answers the questions that real people are actually asking."*

🚆 *From 1,86,074 records — to one powerful answer.* 🚆

</div>
