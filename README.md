# 📉 Customer Churn Prediction System

A machine learning project to predict which customers are likely to leave a service (churn), providing insights for **customer retention strategies**.

---

## 📁 Dataset

* **Source**: Customer churn dataset (telecom or banking sector).
* **Description**: Contains customer information such as demographics, account details, service usage, and churn status (whether the customer left or stayed).

---

## 🎯 Project Overview

* **Task**: Build a model to predict which customers are likely to leave a service.
* **Skills Gained**:

  * 🧩 Classification Modeling
  * 🛠 Feature Engineering
  * 📉 Churn Analysis
  * 📊 Business Insights & Recommendations
* **Deliverable**:

  * Predictive model with churn probabilities
  * Insights on key churn factors
  * Recommendations for customer retention

---

## 🛠 Tools & Libraries Used

* **Python** → `scikit-learn`, `xgboost`, `pandas`, `numpy`
* **Visualization** → `matplotlib`, `seaborn`
* **Platform** → Google Colab

---

## ⚙️ Steps Followed

<details>  
<summary>📌 1. Data Preparation</summary>  

* Loaded churn dataset.
* Handled missing values and categorical variables.
* Encoded features (label encoding / one-hot encoding).
* Scaled numerical features.
* Train-test split for modeling.

</details>  

<details>  
<summary>📊 2. Exploratory Data Analysis (EDA)</summary>  

* Checked churn distribution (class imbalance).
* Analyzed key features affecting churn (e.g., tenure, contract type, monthly charges).
* Visualized churn vs. customer attributes.

</details>  

<details>  
<summary>🤖 3. Model Building</summary>  

* Implemented classification models:

  * Logistic Regression
  * Random Forest
  * XGBoost (best performing)
* Tuned hyperparameters for performance improvement.

</details>  

<details>  
<summary>📏 4. Model Evaluation</summary>  

* Metrics used:

  * Accuracy
  * Precision, Recall, F1-Score
  * ROC-AUC Curve
* XGBoost delivered best balance between accuracy and recall.

</details>  

<details>  
<summary>📊 5. Visualization in Google Colab</summary>  

* Plotted confusion matrix and ROC curve.
* Visualized **feature importance** (top churn factors).
* Created bar/pie charts for churn insights.

</details>  

---

## 📊 Key Results

* ✅ **XGBoost achieved the highest performance** among tested models.
* 🔑 Key churn drivers identified:

  * Contract type (month-to-month contracts have higher churn)
  * Tenure (new customers churn more frequently)
  * Monthly charges (higher bills correlate with churn)
* 🚀 Generated actionable recommendations for customer retention strategies.

---

## 📌 How to Run

### ▶️ Run in Google Colab

1. Upload dataset to Colab.
2. Install dependencies:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost
   ```
3. Run the notebook:

   * Open `Churn_Prediction.ipynb` in **Google Colab**.
   * Execute cells step by step to train models and generate churn insights.

---

✨ This project combines **classification modeling + churn analysis** in Google Colab to deliver predictive insights and actionable recommendations for businesses.

