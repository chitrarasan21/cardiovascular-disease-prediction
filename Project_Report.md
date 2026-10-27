# Cardiovascular Disease Prediction Using Machine Learning

## 1. Introduction

Cardiovascular disease is an important public health problem. In this project I use a patient survey dataset to explore which recorded health and lifestyle measurements are associated with a positive cardiovascular disease label. I then compare five classification methods: Support Vector Machine (SVM), K-Nearest Neighbors (KNN), Decision Tree, Logistic Regression, and Random Forest.

The goal is to demonstrate a machine-learning workflow for a class project. The output is not intended to diagnose a person or replace a health professional.

## 2. Dataset

The file contains age, gender, height, weight, systolic and diastolic blood pressure, cholesterol, glucose, smoking, alcohol, physical activity, and the binary `cardio` outcome. The target is `cardio` (0 = no recorded disease, 1 = disease). The `id` column is only a record identifier, so it is not used as a predictor. Age is recorded in days and is converted to years.

## 3. Data preprocessing

The CSV is semicolon-delimited. Preprocessing removes the identifier and exact duplicate rows, converts age into years, and removes records with clearly implausible height, weight, or blood pressure values. Blood pressure rows with systolic pressure below diastolic pressure are excluded. BMI is calculated from weight and height as an additional feature. Numeric missing values, if present, are filled with the training-set median. Standardization is performed within each model pipeline, so scaling values are learned only from the training data and do not leak from the test set.

## 4. Exploratory analysis and plots

The generated charts include target class balance, distributions of age, height, weight, BMI, and blood pressure by target class, disease proportions across the categorical health and lifestyle variables, and a feature correlation heatmap. `correlation_matrix.csv` and `descriptive_statistics.csv` are included alongside the charts.

When interpreting the figures, a difference between groups is an association in this dataset, not proof that a feature causes cardiovascular disease. Blood pressure and age are continuous measurements, while the cholesterol and glucose variables are ordinal categories.

## 5. Model development

The filtered data is split into training and test sets using an 80:20 stratified split (random state 42). The five requested classifiers are fit using the same split and preprocessing pipeline. Three-fold cross-validation accuracy is also reported for the training portion. The comparison table reports test accuracy, ROC-AUC, and mean cross-validation accuracy. The model with the highest test accuracy is selected for the saved classroom model and confusion matrix.

## 6. Results

After removing implausible values, 68,486 records remained. The classes were close to balanced (about half positive and half negative). On the stratified test set, SVM had the highest accuracy at **73.55%**. Random Forest had a slightly lower accuracy of **73.37%**, but the highest ROC-AUC at **0.798**. Logistic Regression reached **72.99%**, Decision Tree **72.85%**, and KNN **71.42%**. The measured three-fold training cross-validation accuracies are included in `model_comparison.csv`. Since the top two accuracy scores are very close, this split does not prove that SVM is universally the best model.

The selected model for the saved classroom artifact is the SVM because it had the highest test accuracy in this comparison. See `model_comparison.csv`, `classification_report.txt`, `model_accuracy.png`, and `confusion_matrix.png`.

The SVM correctly identified 79% of the negative class (recall) and 68% of the positive disease class on this test split. This shows why accuracy alone does not tell the whole story: some positive cases were missed. The model comparison uses a single fixed holdout, and its results may vary with another split.

## 7. Conclusion

This project demonstrates data cleaning, exploratory visualization, correlation analysis, model comparison, and model saving. The test score is an estimate for a random holdout from this dataset only. Before a model could be considered for health care, it would need external validation, calibration, fairness and safety evaluation, and clinical review.

## 8. Reference

The linked Kaggle notebook was used as a general reference for the topic and the expected machine-learning workflow. The analysis and code here are independently structured around the provided CSV and include explicit preprocessing and evaluation steps: https://www.kaggle.com/code/rahadul231/cardiovascular-disease-prediction-using-ml
