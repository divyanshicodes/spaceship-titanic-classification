# Load Libraries
library(caret)
library(tidyr)
library(dplyr)
library(e1071)
library(mice)
library(nnet)
library(tidyverse)

# Read Data
train <- read.csv("train.csv")
test <- read.csv("test.csv")

# Helper: Mode Function
getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

# Split Cabin into Deck / Num / Side
split_cabin <- function(df) {
  cabin_split <- strsplit(as.character(df$Cabin), "/")
  df$Deck <- sapply(cabin_split, function(x) ifelse(length(x) >= 1, x[1], NA))
  df$Num <- sapply(cabin_split, function(x) ifelse(length(x) >= 2, x[2], NA))
  df$Side <- sapply(cabin_split, function(x) ifelse(length(x) == 3, x[3], NA))
  df$Num <- as.numeric(df$Num)
  return(df)
}

train <- split_cabin(train)
test <- split_cabin(test)

# Impute Missing Values
numeric_cols <- c("Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "Num")
cat_cols <- c("HomePlanet", "Destination", "CryoSleep", "VIP", "Deck", "Side")

for (col in numeric_cols) {
  train[[col]][is.na(train[[col]])] <- mean(train[[col]], na.rm = TRUE)
  test[[col]][is.na(test[[col]])] <- mean(test[[col]], na.rm = TRUE)
}

for (col in cat_cols) {
  train[[col]][is.na(train[[col]])] <- getmode(train[[col]])
  test[[col]][is.na(test[[col]])] <- getmode(train[[col]])
}

# Remove Irrelevant Columns
train$PassengerId <- NULL
test_id <- test$PassengerId
test$PassengerId <- NULL
train$Name <- NULL
test$Name <- NULL

# Convert Categorical to Factor
categorical_cols <- c("HomePlanet", "Destination", "CryoSleep", "VIP", "Deck", "Side", "Transported")
train[categorical_cols] <- lapply(train[categorical_cols], factor)
test[categorical_cols[-7]] <- lapply(test[categorical_cols[-7]], factor)

# Train Validation Split
set.seed(123)
index <- createDataPartition(train$Transported, p = 0.8, list = FALSE)
train_set <- train[index, ]
valid_set <- train[-index, ]

# Normalize Numeric Columns
preprocess_params <- preProcess(train_set, method = c("center", "scale"))
train_norm <- predict(preprocess_params, train_set)
valid_norm <- predict(preprocess_params, valid_set)
test_norm <- predict(preprocess_params, test)

###############################
# 1. KNN Classification
###############################
ctrl <- trainControl(method = "cv", number = 10)
knn_model <- train(Transported ~ ., data = train_norm, method = "knn",
                   trControl = ctrl, tuneLength = 20)

knn_pred <- predict(knn_model, valid_norm)
confusionMatrix(knn_pred, valid_norm$Transported)

knn_test_pred <- predict(knn_model, test_norm)
write.csv(data.frame(PassengerId = test_id, Transported = knn_test_pred),
          "submission_knn.csv", row.names = FALSE)

###############################
# 2. Naive Bayes
###############################
nb_model <- naiveBayes(Transported ~ ., data = train_norm)
nb_pred <- predict(nb_model, valid_norm)
confusionMatrix(nb_pred, valid_norm$Transported)

nb_test_pred <- predict(nb_model, test_norm)
write.csv(data.frame(PassengerId = test_id, Transported = nb_test_pred),
          "submission_nb.csv", row.names = FALSE)

###############################
# 3. Perceptron Neural Network
###############################
nn_model <- nnet(Transported ~ ., data = train_norm,
                 size = 5, decay = 0.1, maxit = 500, trace = FALSE)

nn_pred <- predict(nn_model, valid_norm, type = "class")
confusionMatrix(nn_pred, valid_norm$Transported)

nn_test_pred <- predict(nn_model, test_norm, type = "class")
write.csv(data.frame(PassengerId = test_id, Transported = nn_test_pred),
