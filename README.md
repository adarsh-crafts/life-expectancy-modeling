# Life Expectancy Modeling

This repository contains a data science project analyzing socioeconomic, demographic, and health-related factors influencing life expectancy using WHO data. The project includes a reproducible workflow for preprocessing, exploratory analysis, feature selection, regression modeling, validation, and deployment through a Streamlit application.

## Features

* Cleaned and structured World Health Organization (WHO) life expectancy dataset

* Systematic preprocessing workflow

* Multicollinearity checks and feature selection

* Regression modeling using statistical and machine learning methods

* Evaluation using standard metrics

* Model Validation (coming soon)

* Streamlit dashboard for exploration and prediction

## Project Structure
```
├── app/
│   └── streamlit_app.py
├── data/
├── model/
    └── lifeexp_linreg.pkl
├── notebooks/
    ├── model_development.ipynb
    ├── model_validation.ipynb (coming soon)
    └── EDA.ipynb (coming soon)
├── requirements.txt
└── README.md
```

## Installation
```
pip install -r requirements.txt
```

Run the Streamlit App
```
streamlit run app/app.py
```

## About

This project aims to provide a transparent, reproducible, and interpretable modeling pipeline for understanding key predictors of life expectancy across countries. It is designed with an emphasis on good analytical practice, clarity, and real-world applicability.

A detailed write-up will be available as a blog post soon.

## License

MIT License [READ](LICENSE)