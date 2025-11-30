# Spaceship Titanic Classification in R

This repository contains a full classification pipeline built for the Spaceship Titanic Kaggle dataset.  
The goal of the project is to predict whether a passenger was **Transported** using several supervised learning models implemented in **R**.  

The code file for this project is available separately as **spaceship_classification.R**.

---

## 1. Project Summary

This project was developed as part of a Data Mining course and includes complete preprocessing, feature engineering, model building, evaluation, and Kaggle submission generation.

We compare the performance of four main classification models:

- K Nearest Neighbor  
- Naive Bayes  
- Perceptron Neural Network  
- Support Vector Machine  

The final results show that **SVM achieved the highest accuracy**.

---

## 2. Dataset

You can download the dataset from Kaggle:

- `train.csv`
- `test.csv`

### Main Feature Types
- **Categorical:** HomePlanet, Destination, CryoSleep, VIP, Cabin  
- **Numeric:** Age, RoomService, FoodCourt, ShoppingMall, Spa, VRDeck  
- **Target:** Transported (TRUE or FALSE)

We do not upload Kaggle data here. To run the code, please download files manually and adjust file paths if needed.

---

## 3. Preprocessing Steps

The preprocessing pipeline includes:

1. **Missing value handling**  
   - Numeric features imputed with mean or median  
   - Categorical features imputed with mode  

2. **Cabin feature engineering**  
   Splitting Cabin into:  
   - Deck  
   - Num  
   - Side  

3. **Dropping irrelevant columns**  
   - PassengerId  
   - Name  

4. **Scaling and normalization**  
   Required for KNN, Perceptron, and SVM.

5. **Train and validation split**  
   Stratified 80 to 20 split.

---

## 4. Models Used

### KNN  
- Best K around 17  
- Validation accuracy approx **0.776**  

### Naive Bayes  
- Validation accuracy approx **0.708**

### Perceptron Neural Network  
- Tuned model reaches approx **0.792**

### Support Vector Machine  
- Best performance  
- Validation accuracy approx **0.803**

A submission file is generated for each model.

---

## 5. How to Run the Project

### Requirements  
Install the following R packages:

