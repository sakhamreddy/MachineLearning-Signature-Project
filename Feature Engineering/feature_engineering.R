# Feature Engineering

# Normalizing continuous columns using min-max normalization
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}
SEER_breast_cancer_df[, c(9, 12:14)] <- lapply(SEER_breast_cancer_df[, c(9, 12:14)], normalize)

# Dummy coding and label encoding
Encoded_breast_cancer_df <- as_tibble(predict(dummyVars(~ ., data = SEER_breast_cancer_df, fullRank = TRUE), newdata = SEER_breast_cancer_df))

# Label Encoder
factors <- names(which(sapply(Encoded_breast_cancer_df[, -27], is.factor)))
for (i in factors) {
  encode <- LabelEncoder.fit(Encoded_breast_cancer_df[, i])
  Encoded_breast_cancer_df[, i] <- transform(encode, Encoded_breast_cancer_df[, i])
}
Encoded_breast_cancer_df$Status <- as.factor(Encoded_breast_cancer_df$Status)
