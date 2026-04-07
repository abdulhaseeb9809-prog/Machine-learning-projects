# 🚀 Unified AutoML System (Classification & Regression)

## 📌 Overview

This project is a **Unified AutoML System** that automates the complete machine learning pipeline — from data preprocessing to model selection and prediction.

It allows users to upload any dataset and automatically:

* Detect the problem type (Classification / Regression)
* Preprocess data
* Train multiple models
* Select the best model
* Generate predictions

The system is deployed using **Streamlit** for an interactive user experience.

---

## 🎯 Problem Statement

Building machine learning models requires multiple steps like data cleaning, preprocessing, model selection, and tuning. This process is time-consuming and requires technical expertise.

---

## 💡 Solution

This project solves the problem by creating an **automated ML pipeline** that:

* Handles missing values
* Encodes categorical features
* Scales numerical features
* Trains multiple models
* Performs hyperparameter tuning
* Selects the best-performing model automatically

---

## ⚙️ Features

* 🔍 Automatic problem type detection
* 🧹 Missing value handling (Median / Most Frequent)
* 🔢 Feature encoding using One-Hot Encoding
* ⚖️ Feature scaling using StandardScaler
* 🤖 Model training:

  * Logistic Regression
  * Random Forest
  * XGBoost
* 🎯 Hyperparameter tuning using RandomizedSearchCV
* 🔁 Cross-validation for reliable evaluation
* 📊 Feature importance extraction
* 📁 Upload CSV & get predictions instantly
* 💾 Model download support

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* Streamlit

---

## 🧠 How It Works

1. User uploads a CSV file
2. Selects the target column
3. System identifies problem type
4. Preprocessing pipeline is applied:

   * Missing value handling
   * Encoding
   * Scaling
5. Multiple models are trained
6. Hyperparameter tuning is applied
7. Best model is selected
8. Predictions are generated

---

## ▶️ Run the Application

### Step 1: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Run Streamlit app

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
automl_app/
│
├── app.py
├── automl_code.py
├── requirements.txt
├── README.md
```

---

## 📊 Sample Use Cases

* Income Prediction (Classification)
* Insurance Cost Prediction (Regression)
* Any structured dataset

---

## 🚀 Future Improvements

* Add SHAP for model explainability
* Deploy on cloud (Streamlit Cloud / AWS)
* Add more advanced models
* Improve UI/UX

---

## 👨‍💻 Author

**Abdul Haseeb**

---

## ⭐ Note

This project demonstrates an end-to-end machine learning workflow and is designed to be scalable, efficient, and user-friendly.
