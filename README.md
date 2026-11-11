# WaterGuard: Water Quality Prediction & Potability Classification Pipeline

## Overview
**WaterGuard** is an end-to-end machine learning and data engineering project that predicts whether a water sample is safe for human consumption (`is_safe` / potability) based on physical, chemical, and biological contaminant metrics. 

The project combines data ingestion, anomaly cleaning (handling string artifacts like `#NUM!`), exploratory data analysis (EDA), handling class imbalance using SMOTE, machine learning model comparison, automated testing, and web-based deployment.

---

## Problem Statement
Contaminants such as heavy metals (lead, mercury, arsenic, cadmium), chemical compounds (ammonia, nitrates, perchlorate), and pathogens pose severe public health risks. Manual water quality testing is slow and costly. 

The objective of this project is to build a robust, reproducible data pipeline and an interactive machine learning application that predicts water safety instantly based on lab or sensor parameters.

---

## Dataset Description
* **Source**: IBM / Kaggle Water Quality Prediction Dataset (`waterQualityPrediction.csv`).
* **Dimensions**: 7,999 rows × 21 columns.
* **Target Variable**: `is_safe` (Binary: `1` for safe water, `0` for unsafe/non-potable water).
* **Key Features**:
  - **Heavy Metals & Minerals**: `aluminium`, `arsenic`, `barium`, `cadmium`, `chromium`, `copper`, `lead`, `mercury`, `selenium`, `silver`, `uranium`.
  - **Chemical Compounds**: `ammonia`, `chloramine`, `flouride`, `nitrates`, `nitrites`, `perchlorate`, `radium`.
  - **Biological Indicators**: `bacteria`, `viruses`.

---

## Technologies Used
- **Programming Language**: Python
- **Data Manipulation & Analysis**: Pandas, NumPy
- **Data Visualization**: Matplotlib, Seaborn, Plotly
- **Machine Learning & Resampling**: Scikit-learn, Imbalanced-learn (SMOTE)
- **Model Serialization**: Joblib
- **Web Application**: Streamlit
- **Testing & CI**: Pytest, Git & GitHub

---

## Project Workflow
1. **Data Ingestion**: Load the raw water quality dataset.
2. **Data Cleaning**: Identify and clean string anomalies (e.g., converting `#NUM!` values in the `ammonia` column to numeric floats).
3. **Exploratory Data Analysis**: Analyze distributions, feature correlations, and class imbalances.
4. **Data Preprocessing & Scaling**: Feature standardization and handling class imbalance via SMOTE.
5. **Model Training & Comparison**: Train and evaluate multiple classifiers (Logistic Regression, Decision Tree, Random Forest, KNN, SVM, Gaussian Naive Bayes).
6. **Model Serialization**: Save the best-performing model using Joblib.
7. **Interactive Application**: Build and deploy a Streamlit app for real-time water safety predictions.

---

## Machine Learning Models Compared
The project evaluates several classification algorithms:
* **Logistic Regression**
* **Decision Tree Classifier**
* **Random Forest Classifier**
* **K-Nearest Neighbors (KNN)**
* **Support Vector Machine (SVM)**

### **Evaluation Metrics**:
* Accuracy
* Precision & Recall
* F1-Score
* Confusion Matrix

## Results
The notebooks contain the experiments and model evaluation performed during the project.
The project ultimately explores improvements to the classification models, including experimentation with the Decision Tree approach and handling class imbalance.
For the exact experimental results, metrics, and visualizations, see:
- `water_quality_predicting.ipynb`

---

