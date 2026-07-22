# 🏡 PARCL — AI-Driven Real Estate Buyer Segmentation

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An enterprise-grade Machine Learning project that segments real estate buyers using **unsupervised learning** and provides an interactive Business Intelligence dashboard for customer analysis and investment decision support.

---

# 📌 Project Overview

Real estate companies manage thousands of customers with different demographics, investment behaviors, purchasing goals, and financial capabilities.

Instead of treating every customer the same, this project automatically discovers hidden buyer groups using Machine Learning, allowing companies to better understand their clients and make smarter business decisions.

The project combines:

- Machine Learning
- Customer Segmentation
- Business Intelligence
- Interactive Analytics
- Geographic Visualization
- AI-powered Buyer Prediction

---

# 🎯 Business Objectives

The system helps companies:

- Discover hidden buyer segments.
- Understand customer behavior.
- Analyze investment patterns.
- Compare buyer characteristics.
- Support marketing strategies.
- Improve investment decisions.
- Prioritize high-value customers.
- Predict the segment of new clients using AI.

---

# 🤖 Machine Learning Pipeline

```
Raw Data
      │
      ▼
Data Cleaning
      │
      ▼
Feature Engineering
      │
      ▼
Encoding
(Label Encoding + One-Hot Encoding)
      │
      ▼
Feature Scaling
(StandardScaler)
      │
      ▼
K-Means Clustering
      │
      ▼
Cluster Evaluation
(Elbow Method • Silhouette Score • Davies-Bouldin Index)
      │
      ▼
Hierarchical Clustering Validation
      │
      ▼
PCA Visualization
      │
      ▼
Interactive Streamlit Dashboard
      │
      ▼
AI Buyer Segment Prediction
```

---

# 🧠 Machine Learning Techniques

## Unsupervised Learning

- K-Means Clustering
- Hierarchical Clustering

## Feature Engineering

- Label Encoding
- One-Hot Encoding
- StandardScaler

## Dimensionality Reduction

- Principal Component Analysis (PCA)

## Cluster Validation

- Elbow Method
- Silhouette Score
- Davies-Bouldin Index

---

# 📊 Dashboard Modules

## 📈 Overview

- Executive KPI Cards
- Customer Statistics
- Investment Metrics
- Segment Distribution
- Interactive Charts

---

## 👥 Behavior Analytics

- Buyer Behavior Analysis
- Age Distribution
- Loan Analysis
- Customer Satisfaction
- Investment Behavior

---

## 🌍 Geographic Analytics

- Country Distribution
- Regional Analysis
- Geographic Investment Patterns
- Interactive Maps

---

## 🔍 Segment Insights

Detailed analysis for every buyer segment including:

- Average Investment
- Customer Age
- Satisfaction Score
- Property Ownership
- Segment Characteristics

---

## 📤 Export Center

Export filtered analytical results as:

- CSV
- JSON
- Summary Statistics

---

## 🎯 AI Prediction

Predict the most likely buyer segment for a new client by entering:

- Demographic Information
- Financial Information
- Investment Profile

The prediction engine returns:

- Predicted Buyer Segment
- Confidence Score
- Business Recommendations
- Expected Investment
- Priority Level

---

# 💼 Business Value

This solution enables organizations to:

- Improve customer targeting.
- Optimize marketing campaigns.
- Identify high-value investors.
- Personalize sales strategies.
- Understand geographic markets.
- Support investment planning.
- Make data-driven business decisions.

---

# 📂 Project Structure

```
Real-Estate-Buyer-Segmentation-ML/

│
├── assets/
│
├── data/
│   ├── clients.csv
│   ├── processed_data.csv
│   └── properties.csv
│
├── models/
│   ├── clustered_data.csv
│   ├── kmeans_model.pkl
│   └── scaler.pkl
│
├── reports/
│   └── figures/
│       ├── dendrogram.png
│       ├── elbow_plot.png
│       └── pca_visualization.png
│
├── screenshots/
│
├── app.py
├── main.py
├── utils.py
├── requirements.txt
├── runtime.txt
├── README.md
└── .gitignore
```

---

# 🛠 Technologies

| Category | Tools |
|-----------|----------------|
| Programming | Python |
| Dashboard | Streamlit |
| Machine Learning | Scikit-Learn |
| Data Processing | Pandas |
| Numerical Computing | NumPy |
| Visualization | Plotly |
| Scientific Computing | SciPy |
| Statistical Charts | Matplotlib |

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/yunes-ai-eng/Real-Estate-Buyer-Segmentation-ML.git
```

Move to the project folder

```bash
cd Real-Estate-Buyer-Segmentation-ML
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

# 📷 Dashboard Preview

The project includes interactive dashboards for:

- Executive Overview
- Buyer Behavior Analytics
- Geographic Analysis
- Segment Insights
- Export Center
- AI Buyer Prediction

(Add screenshots inside the `screenshots/` folder.)

---

# 📈 Model Evaluation

The clustering model was evaluated using:

- Elbow Method
- Silhouette Score
- Davies-Bouldin Index
- PCA Visualization
- Hierarchical Dendrogram

These techniques ensure the generated customer segments are meaningful and well-separated.

---

# 👨‍💻 Author

**Yunes Abdulghani Mohammed Ghaleb**

AI & Machine Learning Engineer

Machine Learning Internship Project

PARCL Co. Limited × Unified Mentor

---

# 📄 License

This project is licensed under the MIT License.

---

# ⭐ Acknowledgments

Special thanks to:

- PARCL Co. Limited
- Unified Mentor
- Streamlit
- Scikit-Learn
- Open Source Community

for providing the tools and resources that made this project possible.
