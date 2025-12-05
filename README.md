# Life Expectancy Analysis & Prediction

A data science project analyzing global life expectancy indicators using data from World Health Organization (WHO).  
The goal is to identify key determinants of life expectancy and build an interpretable linear regression model, supported by a complete data science workflow and an interactive Streamlit dashboard.

---

## 1. Project Overview
This project explores how socioeconomic, health, and mortality indicators influence life expectancy across countries.  
Using statistical analysis and machine learning, the project identifies the most impactful predictors and provides an interactive interface for exploration.

---

## 2. Objectives
- Understand global patterns affecting life expectancy  
- Perform data cleaning, EDA, and multicollinearity analysis  
- Build, evaluate, and interpret a linear regression model  
- Deploy a Streamlit app for interactive insights and predictions  

---

## 3. Dataset
**Source:** WHO Life Expectancy Dataset  
**Rows:** 2,930  
**Features:** 22 indicators including mortality, immunization, economic metrics, and health factors.

---

## 4. Methodology
- Exploratory Data Analysis (EDA)  
- Correlation & Multicollinearity (VIF)  
- Feature Selection (Forward Selection)  
- Scaling & Preprocessing  
- Linear Regression modeling  
- Regression diagnostics (residuals, Q-Q, homoscedasticity)  
- Performance evaluation (MAE, MSE, RMSE, R²)

---

## 5. Key Insights
- Adult mortality, HIV incidence, schooling, and BMI strongly influence life expectancy  
- Immunization and nutrition indicators show impactful relationships  
- Multicollinearity reduced by removing highly correlated variables  

*(Replace with your own final insights.)*

---

## 6. Streamlit App
An interactive dashboard to:  
- Explore global indicators  
- Visualize correlations  
- View model results  
- Predict life expectancy  

Run locally:
```bash
streamlit run app.py
s