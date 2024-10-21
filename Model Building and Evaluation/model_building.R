# Model Building and Evaluation

# Splitting data into training and test sets
set.seed(123)
without_smote <- createDataPartition(Encoded_breast_cancer_df$Status, p = 0.8, list = FALSE)
train_breast_cancer <- Encoded_breast_cancer_df[without_smote, ]
test_breast_cancer <- Encoded_breast_cancer_df[-without_smote, ]

# Balancing the training set using SMOTE
train_breast_cancer_smote <- DMwR::SMOTE(Status ~ ., data = as.data.frame(Encoded_breast_cancer_df[without_smote, ]), perc.over = 200)

# Model: Support Vector Machine
svm_model_bal <- ksvm(Status ~ ., data = train_breast_cancer_smote, kernel = "vanilladot")
fit_svm_bal <- predict(svm_model_bal, test_breast_cancer)
svm_cm_bal <- confusionMatrix(table(fit_svm_bal, test_breast_cancer$Status), positive = "0")

# Model: Decision Trees
dt_bal <- C5.0(train_breast_cancer_smote[, -27], train_breast_cancer_smote$Status)
fit_dt_bal <- predict(dt_bal, test_breast_cancer)
dt_cm_bal <- confusionMatrix(table(fit_dt_bal, test_breast_cancer$Status), positive = "0")

# Model: Random Forests
set.seed(100)
rf_model_bal <- train(Status ~ ., data = train_breast_cancer_smote, method = "rf", metric = c("Accuracy"), trControl = trainControl(method = "cv", number = 10, selectionFunction = "oneSE"))
predict_rf_bal <- predict(rf_model_bal, newdata = test_breast_cancer)
confusionMatrix(table(predict_rf_bal, test_breast_cancer$Status))
