# 🎓 Student Academic Performance Analytics

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Big Data Analytics Case Study: Transforming Student Academic Records into Early Warnings & Prescriptive Interventions.**

---

## 👥 Project Team
- **U. Sampath**
- **P. Kartheek**
- **T. Abhishek**

---

## 📌 Executive Summary
Educational institutions frequently suffer from delayed visibility into struggling students until semester exams conclude. This interactive Big Data Analytics platform operationalizes academic records across four analytical pillars:

1. **Descriptive Analytics**: Real-time performance dashboards, grade distribution, and attendance profiles.
2. **Diagnostic Analytics**: Root cause identification linking attendance thresholds and assessment dips to academic outcomes.
3. **Predictive Analytics**: Machine learning early-warning classification models (Logistic Regression & Random Forest) with stratified k-fold cross-validation and ROC-AUC evaluation.
4. **Prescriptive Analytics**: Automated, student-tailored academic interventions (counseling, remedial classes, assignment deadlines).

---

## 🛠️ Tech Stack
- **Dashboard Framework**: [Streamlit](https://streamlit.io/)
- **Machine Learning**: [scikit-learn](https://scikit-learn.org/) (Logistic Regression, Random Forest, Imputers, Scalers)
- **Data Manipulation**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Interactive Visualizations**: [Plotly Express & Graph Objects](https://plotly.com/python/)

---

## 🚀 Quickstart Guide

### 1. Clone & Setup
```bash
git clone https://github.com/Kartheekputti/student-academic-performance-analytics.git
cd student-academic-performance-analytics
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Locally
```bash
streamlit run app.py
```
The app will automatically open in your default browser at `http://localhost:8501`.

---

## 📊 Features & Workflow
- **Custom CSV Upload & Synthetic Data**: Upload your own institutional dataset or explore with the built-in synthetic benchmark.
- **Automated Data Cleaning**: Cleans missing values, validates score bounds, maps schema aliases, and flags duplicate student profiles.
- **Interactive Predictive Thresholds**: Dynamically adjust pass marks and classification thresholds to balance model sensitivity and specificity.
- **Exportable Interventions**: Export flagged student intervention lists directly to CSV for administrative and mentoring follow-ups.
