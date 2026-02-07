# 📊 First-ML: Model Comparison Using Machine Learning

This repository contains a beginner-friendly machine learning project where multiple regression models are implemented and compared on the same dataset "Delaney Solubility with Descriptors".
The goal is to **understand model performance differences** between classical and ensemble learning methods.

---

## 🚀 Project Overview

In this project, we:

* Implement **Linear Regression** as a baseline model
* Implement **Random Forest Regressor** as a non-linear ensemble model
* Use a **separate comparison script** to fairly evaluate both models
* Compare models using standard regression metrics

This project also demonstrates **good ML engineering practices** such as:

* Modular code design
* Reusable training functions
* Centralized evaluation pipeline

---

## 🧠 Models Used

* **Linear Regression**
* **Random Forest Regressor**

---

## 📈 Evaluation Metrics

The models are compared using:

* **Mean Squared Error (MSE)**
* **R² Score**

These metrics help evaluate both prediction accuracy and goodness of fit.

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/pranjal25r/First-ML.git
cd First-ML
```

### 2️⃣ Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux / Mac
venv\Scripts\activate           # Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run the Project

Run the model comparison script:

```bash
python model_comparison.py
```

This will:

* Load and preprocess the dataset
* Train both models
* Print a comparison table of evaluation metrics

---


## 🔮 Future Improvements

* Add more models (Ridge, Lasso, XGBoost)
* Hyperparameter tuning using GridSearchCV
* Cross-validation
* Visualization of predictions
* Model persistence using `joblib`

---

## ⭐ If You Like This Project

Feel free to ⭐ the repository and use it as a reference for:

* ML coursework
* Mini projects
* Internship preparation