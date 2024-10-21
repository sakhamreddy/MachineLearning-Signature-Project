# Ensemble Learning with Bagging and Majority Voting

# Bagging
RNGversion("3.5.2")
set.seed(300)
bag_bal <- bagging(Status ~ ., data = train_breast_cancer_smote, nbag = 25)
bag_pred_bal <- predict(bag_bal, test_breast_cancer)
confusionMatrix(table(bag_pred_bal, test_breast_cancer$Status))

# Majority Voting
predict_status <- function(testdata) {
  svm_pred <- predict(svm_model_bal, testdata)
  log_reg_pred <- predict(log_bal, testdata)
  dec_tree_pred <- predict(dt_bal, testdata, type = "class")
  bag_pred <- predict(bag_bal, testdata)
  
  # Using majority voting
  pred_majority <- ifelse(sum(svm_pred == "1") + sum(log_reg_pred == "1") + sum(dec_tree_pred == "1") + sum(bag_pred == 1) > 1, "1", "0")
  return(pred_majority)
}

# Making predictions using majority voting
predict_status(test_breast_cancer)
