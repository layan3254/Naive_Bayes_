# 🚢 Titanic Survival Prediction using Naive Bayes

This project explores the **Titanic dataset** to predict passenger survival using three variants of the **Naive Bayes** algorithm. We focused on data preprocessing, exploratory analysis, and advanced hyperparameter tuning to achieve optimal performance.


## 🎯 Project Overview

We implemented three versions of Naive Bayes to compare how they handle different data distributions:

1.  **Gaussian NB:** For continuous numerical data (Age, Fare).
2.  **Multinomial NB:** For discrete/binned data.
3.  **Bernoulli NB:** For binary features.

## 🧬 Key Features

We selected features with the highest historical correlation to survival:

  - **Sex:** The strongest indicator.
  - **Pclass:** class difference (1st class safer)
  - **Pclass & Fare:** Indicators of socio-economic status.

## 📊 Exploratory Data Analysis (EDA)

Our analysis confirmed key survival trends:

  - **Gender Gap:** Females had a significantly higher survival rate.
  - **Socio-economic Factor:** Passengers in 1st Class were prioritized.
  - **Data Distribution:** We identified rare values and outliers in Age and Fare that required special handling.

| Feature | Correlation with Survival |
| :--- | :--- |
| Sex | Very High |
| Pclass | High |
| Fare | Moderate |

## 💡 Solution (Laplace Smoothing)

A major challenge in Naive Bayes is the **Zero Probability Problem**. If a model encounters a rare feature value it hasn't seen during training, the probability becomes zero, "blinding" the model to all other evidence.

**Our Solution:**
We applied **Laplace Smoothing (Alpha)**. By adding a small weight ($\alpha$) to all features, we ensured the model maintains a global perspective and never reaches a zero probability, significantly improving robustness.

## 🚀 Model Performance

We used `GridSearchCV` to find the best hyperparameters. **Multinomial NB** and **Gaussian NB** emerged as the top performer.

| Model | Accuracy |
| :--- | :--- |
| **Bernoulli NB** | **73%** |
| **Multinomial NB** | **78%** |
| **Gaussian NB** | **78%** |

## 🛠️ Tools & Libraries

  - **Language:** Python
  - **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
