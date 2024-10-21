# Data Acquisition and Cleaning

# Data imported from the local folder and read it using read.csv function and
# set parameter "stringAsFactors" to "TRUE" to convert the character features into factor levels

# Data set links
# https://ieee-dataport.s3.amazonaws.com/open/7249/SEER%20Breast%20Cancer%20Dataset%20.csv?response-content-disposition=attachment%3B%20filename%3D%22SEER%20Breast%20Cancer%20Dataset%20.csv%22&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAJOHYI4KJCE6Q7MIQ%2F20230426%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230426T144022Z&X-Amz-SignedHeaders=Host&X-Amz-Expires=86400&X-Amz-Signature=8c9345b83cd8541437e52b807879d9c6b08a16cc764c21a494a7b0230e9fdad0
# https://www.kaggle.com/datasets/reihanenamdari/breast-cancer?select=Breast_Cancer.csv

ID <- "17XYkkiYNGdp5sY15qNIIsFk04YO_n9IK"

seer_data <- read.csv(sprintf("https://docs.google.com/uc?id=%s&export=download", ID))
head(seer_data)

# Removing extra column containing NA values (duplicate column)
SEER_data <- seer_data[,-4]

# SEER cancer data containing 4024 rows and 14 independent columns and 1 dependent column (target variable)
dim(SEER_data)
str(SEER_data)

# Changing column names for convenience and easy to understand
colnames(SEER_data)[1:15] <- c("Age", "Race", "Marital_Status", "T_stage", "N_stage", "sixth_stage", "Grade",
  "A_stage", "Tumor_size", "Estrogen_status", "Progesterone_status",
  "Regional_nodes_examined", "Regional_nodes_positive", "Survival_months", "Status")

# Imputation of missing values
# Replacing large number of missing values in the columns with their respective means and removing low volume of missing values
SEER_data$Regional_nodes_positive[is.na(SEER_data$Regional_nodes_positive)] <- mean(SEER_data$Regional_nodes_positive, na.rm = TRUE)
SEER_data$Tumor_size[is.na(SEER_data$Tumor_size)] <- mean(SEER_data$Tumor_size, na.rm = TRUE)
SEER_data$Regional_nodes_examined[is.na(SEER_data$Regional_nodes_examined)] <- mean(SEER_data$Regional_nodes_examined, na.rm = TRUE)

# Removing rows with missing values in the data
SEER_breast_cancer_df <- na.omit(SEER_data)
