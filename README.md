# Telecom Customer Churn Prediction
![Workflow Diagram](Workflow.jpeg)

## Overview

**Telecom Customer Churn Prediction** is a machine learning system designed to identify telecom customers who are likely to stop using services.

The system processes historical customer behavior data, performs data cleaning and exploratory analysis, engineers behavioral features, selects the most relevant predictors, trains multiple machine learning models, and identifies high-risk customers for targeted retention.

The project defines churn as customers having **zero recharge and zero data usage in month nine** and removes month-nine features to prevent data leakage.

---

## Aim

- Predict customers who are likely to churn
- Identify the key behavioral factors associated with churn
- Segment customers according to their churn risk
- Provide actionable insights for customer retention
- Build an interpretable machine learning solution using SHAP analysis

---

## Key Features

- **Data Cleaning** — Handles missing values and removes data leakage
- **EDA & Behavioral Analysis** — Identifies trends, relationships, outliers, and customer segments
- **Feature Engineering** — Creates activity, RFM, trend, frequency, and efficiency-based features
- **Feature Selection** — Reduces the feature space from 150 to 25 important features
- **Class Balancing** — Uses SMOTE to address the 14.5% churn rate
- **Multi-Model Training** — Compares Random Forest, XGBoost, LightGBM, CatBoost, and Logistic Regression
- **Model Evaluation** — Uses accuracy, precision, recall, F1, ROC-AUC, and cumulative gains
- **SHAP Explainability** — Identifies the most important drivers behind churn predictions
- **Risk Segmentation** — Classifies customers into low, medium, and high churn-risk groups
- **Retention Insights** — Converts model findings into actionable customer retention recommendations

---

## Benefits

- **Early Churn Detection** — Identifies high-risk customers before they leave
- **Targeted Retention** — Helps focus retention efforts on customers with higher churn probability
- **Better Customer Understanding** — Reveals behavioral patterns associated with churn
- **Explainable Predictions** — SHAP analysis explains why customers are classified as high risk
- **Reduced Manual Analysis** — Automates customer risk identification
- **Data-Driven Decisions** — Supports targeted retention strategies using customer behavior

---

## Workflow

The system follows this overall process:

**Raw Telecom Data → Data Cleaning → EDA → Train/Test Split → Feature Engineering → Feature Selection & SMOTE → Model Training → Model Evaluation → SHAP Explainability → Risk Segmentation → Retention Insights**

### Workflow Steps

**1. Data Collection**  
Raw telecom customer data is loaded into the system.

**2. Data Cleaning**  
The churn target is created using month-nine behavior. Month-nine columns are removed to prevent data leakage, while missing numerical and categorical values are handled using median and mode respectively.

**3. Exploratory Data Analysis**  
Customer behavior is analyzed through skewness transformation, outlier capping, correlation analysis, categorical association testing, and customer segmentation.

**4. Train/Test Split**  
The dataset is divided into **80% training and 20% testing** using stratification to preserve the churn distribution.

**5. Feature Engineering**  
Behavioral features such as total activity, RFM metrics, frequency, activity trends, usage variability, and efficiency ratios are created.

**6. Feature Selection & Class Balancing**  
The feature set is reduced from **150 to 25 features** using VIF, correlation filtering, and Random Forest importance. SMOTE is applied to address class imbalance.

**7. Model Training**  
Five machine learning models are compared using five-fold cross-validation and optimized for F1 score.

**8. Model Evaluation**  
Models are evaluated using accuracy, precision, recall, F1 score, ROC-AUC, precision-recall curves, confusion matrices, and cumulative gains.

**9. SHAP Explainability**  
SHAP analysis identifies the features contributing most strongly to individual and overall churn predictions.

**10. Risk Segmentation & Retention Insights**  
Customers are categorized into low, medium, and high-risk groups, and the major churn drivers are converted into actionable retention recommendations.

---

## Model Results

The **Random Forest** model achieved the best overall performance:

- **Accuracy:** 86.42%
- **Precision:** 84.23%
- **Recall:** 82.07%
- **F1 Score:** 0.8314
- **AUC-ROC:** 0.91

The confusion matrix contains:

- **True Negatives:** 2,845
- **False Positives:** 412
- **False Negatives:** 389
- **True Positives:** 1,834

The cumulative gains analysis shows that targeting the **top 20% highest-risk customers captures more than 65% of actual churners**. :contentReference[oaicite:1]{index=1}

---

## Top Churn Drivers

SHAP analysis identified the following major churn drivers:

1. **Total Activity** — Impact: 0.0842
2. **Activity Trend** — Impact: 0.0765
3. **Month 8 Recharge Amount** — Impact: 0.0718
4. **Frequency of Interactions** — Impact: 0.0684
5. **Usage per Activity Ratio** — Impact: 0.0653

These insights support targeted actions such as early intervention for declining engagement, loyalty rewards for low-recharge customers, increased customer touchpoints, and service-quality improvements. :contentReference[oaicite:2]{index=2}

---

## Use Cases

- Telecom customer churn prediction
- Customer retention
- High-risk customer identification
- Targeted retention campaigns
- Customer behavior analysis
- Churn driver analysis
- Customer risk segmentation
- Telecom business intelligence

---

## Project Goal

The goal of **Telecom Customer Churn Prediction** is to transform customer behavioral data into predictive churn intelligence that helps telecom businesses identify high-risk customers and take targeted retention actions.

> **Customer Data → Analyze Behavior → Engineer Features → Predict Churn → Explain Risk → Segment Customers → Take Retention Action**

---

## Final Output

The project produces:

- A trained **Random Forest churn prediction model**
- Customer churn risk predictions
- Low, medium, and high-risk customer segments
- SHAP-based churn explanations
- Churn driver analysis
- Model evaluation metrics
- Professional visualizations including ROC curves, confusion matrices, cumulative gains, and risk segmentation charts

The trained model is saved as `random_forest_churn_model.pkl` and SHAP values are saved as `shap_values.pkl`. The project contains **38+ visualization plots** documenting model performance and customer risk insights. :contentReference[oaicite:3]{index=3}

---

## Workflow Diagram

![Telecom Customer Churn Prediction Workflow](docs/workflow.png)
