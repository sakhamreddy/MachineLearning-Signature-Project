---
title: "Signature Project"
output:
  pdf_document: default
  html_notebook: default
---

#Required packages
```{r,warning=FALSE,message=FALSE}
#Imported required packages to perform the project
library(corrplot)
library(magrittr)
library(dplyr)
library(ggcorrplot)
library(psych)
library(RVAideMemoire)
library(moments)
library(tidyverse)
library(CatEncoders)
library(DMwR)
library(kernlab)
library(C50)
library(gmodels)
library(caret)
library(Metrics)
library(irr)
library(plotly)
library(cvms)
library(ipred)
library(caretEnsemble)
```
#Data Acquisition
```{r}
#Data imported from the local folder and read it using read.csv function and
#set parameter "stringAsFactors" to "TRUE" to convert the character features into
#factor levels

#Data set links
#https://ieee-dataport.s3.amazonaws.com/open/7249/SEER%20Breast%20Cancer%20Dataset%20.csv?response-content-disposition=attachment%3B%20filename%3D%22SEER%20Breast%20Cancer%20Dataset%20.csv%22&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAJOHYI4KJCE6Q7MIQ%2F20230426%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230426T144022Z&X-Amz-SignedHeaders=Host&X-Amz-Expires=86400&X-Amz-Signature=8c9345b83cd8541437e52b807879d9c6b08a16cc764c21a494a7b0230e9fdad0
#https://www.kaggle.com/datasets/reihanenamdari/breast-cancer?select=Breast_Cancer.csv

ID <- "17XYkkiYNGdp5sY15qNIIsFk04YO_n9IK"

seer_data <- read.csv(sprintf("https://docs.google.com/uc?id=%s&export=download", ID))
head(seer_data)

#Removing extra column containing NA values(duplicate column)
SEER_data <- seer_data[,-4]


#Seer cancer data containing 4024 rows and 14 independent columns and 1 
#dependent column(target variable)
dim(SEER_data)
#string output showing the factor levels and integer columns
str(SEER_data)


summary(SEER_data)
#summary output showing that data set does not containing any NA values and I
#think that there is not much difference between the min-max values of the 
#integer columns

#changing column names for convenience and easy to understand
colnames(SEER_data)[1:15] <- c("Age","Race","Marital_Status","T_stage","N_stage","sixth_stage","Grade",
  "A_stage","Tumor_size","Estrogen_status","Progesterone_status",
  "Regional_nodes_examined","Regional_nodes_positive","Survival_months","Status")

```


#Data Exploration(EDA)
```{r}
#Exploratory data analysis using histograms 
#Encoded the target feature Status(Alive=0, Dead=1)

SEER_data$Status <- as.character(SEER_data$Status)
class(SEER_data$Status)

#Dummy coding the target variable and assigned Alive to "0" and Dead to "1"
SEER_data$Status[SEER_data$Status=="Alive"] <- 0
SEER_data$Status[SEER_data$Status=="Dead"] <- 1

#Changing the class from character to binary(factor)
SEER_data$Status <- as.factor(SEER_data$Status)


#Factor analysis was done by made bar plots of categorical variables to target 
#variable to know how different levels in the categorical variables response to 
#the target variable 
#Created a list to present the in simple code

cat_variables <- list("Race","Marital_Status","T_stage","N_stage","sixth_stage",
                      "Grade","A_stage","Estrogen_status","Progesterone_status")

#created a for loop to display all the bar plots at once with relation to
#target variable

par(mfrow=c(3,3))
for (i in cat_variables){
 gg_plot <-  ggplot(SEER_data, aes_string(x = i, fill = SEER_data$Status))+ 
    geom_bar( stat = "count")+ scale_fill_discrete(name = "Status")+geom_text(aes(label=paste(after_stat(round(count / sum(count) * 100,1))
                                                                                      ,"%")),
     stat='count',
     nudge_y=0.125)
 print(gg_plot)
}

paste0("Analysis:From the bar plots, it is very clear that most of the data containing 
white married woman(71% and 57%). Where black and other races contribute to very 
low percentage. Similarly, separated marital status women are very less compared 
to other marital status people.Most of the patients having T1,T2 and N1 stages. 
It is indicating that women developing cancer tumor having low size and it's not 
wide spread to other areas near the chest.Remaining 1.5% women having T4 stage 
and i think those poeple having large tumor with size of > 5cm. Those are most 
probably fail the regional_nodes_positive test, because it spread to other organs 
in T4 stage. Here one more intersting fact I observed that T4 stage poeple Alive
to dead ratio was 50-50.N1 stage indicated that cancer has spread to near by 1-3 
auxillary lymph nodes indicating micrometastasis.As like T4 stage, N3 stage has 
also similar ratio to alive and dead patients.Sixth stage groups IIA,IIB and 
IIIA constitute majority of the data and same fact repeated here that IIIC stage 
people having nearly 60-40% survival rates.Regarding Grade, it might depends on 
the above factors as well as estrogen,progesterone levels for each patient. But, 
those patients having tumor well differentiated having low survival rates and
majority of the patients were in GradeII(moderately differentiated).Remaining 
data features A-stage(Regional), Estrogen(Positive), Progesterone(Positive) 
containing most of the patients.")


#To know the relationship between the continuous variables to target variable
#using byf.hist function to produce dense plots
par(mfrow=c(1,1))
hist(SEER_data$Age)
byf.hist(Age~Status, data = SEER_data)
par(mfrow=c(2,2))
byf.hist(Tumor_size~Status, data=SEER_data)
byf.hist(Regional_nodes_examined~Status, data=SEER_data)
byf.hist(Regional_nodes_positive~Status, data=SEER_data)
byf.hist(Survival_months~Status, data=SEER_data)



#Detection of outliers which are present in the continuous columns. Here I 
#used graphical representation box plot to find the outliers in the continuous
#columns
set.seed(123)
par(mfrow=c(1,1))
boxplot(SEER_data[,c("Age","Tumor_size",
  "Regional_nodes_examined","Regional_nodes_positive","Survival_months")])

#Output"From the plot, it is obvious that columns tumor size,
#regional nodes examined,regional nodes positive and survival months containing 
#outliers.It is clear that column regional nodes positive containing large number of #outliers. I had replaced them with NA values with below code

for (i in c("Age","Tumor_size",
  "Regional_nodes_examined","Regional_nodes_positive","Survival_months"))
{
  value = SEER_data[,i][SEER_data[,i] %in% boxplot.stats(SEER_data[,i])$out]
  SEER_data[,i][SEER_data[,i] %in% value] = NA
} 

#checking the outlier values are replaced with NA values using sum and is.na
#functions. Both tumor size and regional_nodes_positive containing high
#volume of NA values

sum(is.na(SEER_data$Tumor_size))
sum(is.na(SEER_data$Regional_nodes_examined))
sum(is.na(SEER_data$Regional_nodes_positive))
sum(is.na(SEER_data$Surival_months))


#correlation
#Here I find correlation using one-hot encoding using model.matrix and plot
#the relations between each variables
model.matrix(~0+., data=SEER_data) %>% 
  cor(use="pairwise.complete.obs") %>% 
  ggcorrplot(show.diag=FALSE, type="lower", lab=TRUE, lab_size=2)

#Using another, most commonly used method to find the correlation as well as
#distribution of the data is pairs.panels from the "psych" package

pairs.panels(SEER_data[,c("Age","Race","Marital_Status","T_stage","N_stage",
"sixth_stage","Grade","A_stage","Tumor_size","Estrogen_status","Progesterone_status",
"Regional_nodes_examined","Regional_nodes_positive","Survival_months","Status")])

```
output:It is obvious from these pairs.panels that there is 
multicollinearity between the independent variables. 6th_stage and N-stage 
have a very strong positive correlation.N-stage, 6-stage, tumor size,positive 
regional nodes, nodes examined are all interrelated with each other.

Evaluation of Distribution
From the above pairs.panels plot, it is concluded that the continuous columns 
are distributed differently(skew in the distribution)
Near Normal-distribution: Regional_nodes_examined
Right-skew-distribution: Regional_nodes_positive, Tumor_size
Left_skew_distribution: Survival_months, Age

From the above conclusions, it is mandatory to perform transformation(log,
inverse) or standardization of data




#Data cleaning & shaping

##Identify missing values
```{r}

#The above data set containing large volume of NA values and we will identify
#them by using is.na() function

sum(is.na(SEER_data))

#We can observe that the entire data set has significant missing values.
#Therefore, we must replace the missing data in the columns with their 
#respective means.

```


#Imputation of data
```{r}
#Replacing large number of missing values in the columns
#with their respective means and survival months containing low volume of 
#missing values. So, I decided to remove them instead of keeping them

SEER_data$Regional_nodes_positive[is.na(SEER_data$Regional_nodes_positive)] <- mean(SEER_data$Regional_nodes_positive, na.rm = TRUE)
SEER_data$Tumor_size[is.na(SEER_data$Tumor_size)] <- mean(SEER_data$Tumor_size, na.rm = TRUE)
SEER_data$Regional_nodes_examined[is.na(SEER_data$Regional_nodes_examined)] <- mean(SEER_data$Regional_nodes_examined, na.rm = TRUE)

#Removed small volume of missing values in the data column 
SEER_breast_cancer_df <- na.omit(SEER_data)


#checking if there any missing values in the data
sum(is.na(SEER_breast_cancer_df))
#So, now we have zero missing values in the data set. We move forward with 
#standardization techniques

```

#Distribution checking
```{r}
#summary stats showing that there is min-max value difference in both tumor_size
#and survival months column
summary(SEER_breast_cancer_df)
par(mfrow=c(2,2))
hist(SEER_breast_cancer_df$Age)
hist(SEER_breast_cancer_df$Tumor_size)
hist(SEER_breast_cancer_df$Survival_months)
hist(SEER_breast_cancer_df$Regional_nodes_examined)
hist(SEER_breast_cancer_df$Regional_nodes_positive)
#From this hist plot, regional_nodes_positive data mostly exist in 1
table(SEER_breast_cancer_df$Regional_nodes_positive)
```

#Transformation
```{r}
set.seed(123)
#From the above histograms, we have seen that data does not distributed 
#normally. We can visually depict that from the above hist plots.So, we need to
#perform transformation(log, sqrt, inverse) for continuous variables to remove
#skewness in the data to perform the model training.
#So standardization does not remove the skewness the data.That's why I chose
#different transformation parameters for different skew's in the data. I took
#these from USCS

#Standardization
#performed min-max normalization on the continuous columns
normalize <- function(x, na.rm = TRUE) {
    return((x- min(x)) /(max(x)-min(x)))
}
SEER_breast_cancer_df[,c(9,12:14)] <- lapply(SEER_breast_cancer_df[,c(9,12:14)],
                                             normalize)
head(SEER_breast_cancer_df)

#Transformation
#From the above plots, columns regional_nodes_positive showing heavy right skew
#and survival column showing moderate right skew
SEER_breast_cancer_df$Regional_nodes_positive <- sqrt(SEER_breast_cancer_df$Regional_nodes_positive)
SEER_breast_cancer_df$Tumor_size <- sqrt(SEER_breast_cancer_df$Tumor_size)


#Comparatively, the normality violation decreased from original data. The 
#coefficient values for both Age and regional_nodes_examined features increased
#with transformation. That's the reason I didn't transform them
skewness(SEER_breast_cancer_df$Tumor_size, na.rm = TRUE)
skewness(SEER_breast_cancer_df$Regional_nodes_examined, na.rm = TRUE)
skewness(SEER_breast_cancer_df$Regional_nodes_positive, na.rm = TRUE)
skewness(SEER_breast_cancer_df$Survival_months, na.rm = TRUE)
skewness(SEER_breast_cancer_df$Age, na.rm = TRUE)


#Skewness results before transformation
#1.016867
#0.2927328
#1.628163
#-0.5419102
#-0.2190115

#Skewness results after transformation
#0.2895557
#0.2927328
#0.3988236
#-0.5419102
#-0.2190115

```
We got different skewness coefficients and perform transformations accordingly
Here, For 
For Right-skew: log,sqrt,inverse
For Left-skew: squares, cubes
For Normal-distribution: No parameter required(sqrt)-moderate

For variables with high normality violation value even positive or negative, we
should perform inverse transformation. For large violation, use log transformation
and for moderate violation we should use sqrt transformation
Here
Tumor_size(0.91)-sqrt transformation
Regional_nodes_examined(0.29)- sqaure root transformation
Regional_nodes_positive(1.27)- Inverse transformation
Survival_months(-0.54)-log transformation

Note:Here we also check for the linearity and heteroscedasticity, when dependent
and independent variables are directly proportional or exhibiting positive
correlation, we will first consider "log" transformation.And when they
exhibiting negative correlation, we should consider "sqrt" transformation
first

#dummy coding
```{r}
#Assigning the transformed data to Encoded_breast_cancer_df 
Encoded_breast_cancer_df <- SEER_breast_cancer_df

#using dummyvars from the caret package to perform the dummy coding
Encoded_breast_cancer_df <- as_tibble(predict(
  dummyVars( ~ ., data = Encoded_breast_cancer_df, fullRank = TRUE), newdata = Encoded_breast_cancer_df))
#Encoded_breast_cancer_df[,c(2:8,10:11)] <- lapply(Encoded_breast_cancer_df[,c(2:8,10:11)], factor)
head(Encoded_breast_cancer_df)

factors <- names(which(sapply(Encoded_breast_cancer_df[,-27], is.factor)))

# Label Encoder
for (i in factors){
  encode <- LabelEncoder.fit(Encoded_breast_cancer_df[, i])
  Encoded_breast_cancer_df[, i] <- transform(encode, Encoded_breast_cancer_df[, i])
}
colnames(Encoded_breast_cancer_df)[27] <- "Status"
Encoded_breast_cancer_df$Status <- as.factor(Encoded_breast_cancer_df$Status)

```


#Principal Component Analysis(PCA)
```{r}
#Assigning data set to pca_data
pca_data <- Encoded_breast_cancer_df

#Applying principal component analysis using prcomp from stats on the continuous
#columns
pca_comp <- prcomp(pca_data[,c(1,21,24:26)], center = TRUE)

#summary of the principal components
summary(pca_comp)

#First component explains 91% variability and remaining showing similar variability
pca_comp$sdev ^ 2
print(pca_comp$rotation)

#From this pca data, all the variables contribute similarly in different principal
#components.So, I decided to choose all the variables in training the model.
```
output: From this pca data, all the variables contribute similarly in different 
principal components.So, I decided to choose all the variables in training the 
model.



#Feature Engineering
```{r}
cor_Seer <- cor(Encoded_breast_cancer_df[,c(1,21,24:26)])
cor_Seer
corrplot(cor_Seer)
#From the corr plot, we have seen that variables tumor_size and 
#regional_nodes_positive are positively correlated and regional_nodes_examined
#and regional_nodes_positive are correlated with each other. Feature 
#survival_months and age both have no relationship with other variables.

#Need to find correlation between race & marital status to tumor size and regional
#nodes positive

```


#Splitting data
```{r}
#For the given data set, I choose 80% training and 20% validation data set
#splitting
#Imbalanced data splitting
set.seed(123)
without_smote <- createDataPartition(Encoded_breast_cancer_df$Status, p=0.8, 
                                     list=FALSE)

train_breast_cancer <- Encoded_breast_cancer_df[without_smote,]
prop.table(table(train_breast_cancer$Status))
test_breast_cancer <- Encoded_breast_cancer_df[-without_smote,]


#splitting the data and balancing the train data set which containing 
#difference in dependent variable
train_breast_cancer_smote <- DMwR::SMOTE(Status~., data=as.data.frame(Encoded_breast_cancer_df[without_smote,]), perc.over = 200)
prop.table(table(train_breast_cancer_smote$Status))

#Here, we see that train data before and after balancing with smote function.
#Now,we perform the models using both imbalanced and balanced training data
```
#selecting models for data set
```{r}
#It’s actually one of the difficult task to select the appropriate algorithm for
#specific data and depends on circumstances we need. Here, I decided to go 
#through the some of classification algorithms such as Logistic Regression, 
#Decision Trees, Support Vector Machine (SVM), Random Forest (RF). It’s all 
#about #trial-and-error process and finally compare the all the models by specific 
#parameters and concluded to one model. Particularly, I chose above because my 
#data set containing both categorical and continuous variables and my target 
#variable is categorical(binary).
```



#Model1(support vector machines)#Imbalanced data(Model Training)
```{r}
set.seed(123)
#Performing SVM(Support Vector Machines) algorithm on imbalanced training data
svm_model_imb <- ksvm(Status~., data=train_breast_cancer, kernel="vanilladot")

#Evaluating model on test data(unseen data)
fit_svm_imb <- predict(svm_model_imb, test_breast_cancer)

#Table to calculate the Accuracy, precision, recall and F-scores
tab_imb <- table(fit_svm_imb, test_breast_cancer$Status)
svm_cm_imb <- confusionMatrix(tab_imb, positive = "0")
svm_cm_imb
table(test_breast_cancer$Status)
#Results
print(paste("For Imbalanced data:", "Precision is:",caret::precision(tab_imb),
"Recall is:",sensitivity(tab_imb),"F-score is:",caret::F_meas(tab_imb)))

```


#Model1(Support Vector Machines)#Balanced data(Model Training)
```{r}
set.seed(123)
#Performing SVM(Support Vector Machines) algorithm on balanced training data
svm_model_bal <- ksvm(Status~., data=train_breast_cancer_smote, kernel="vanilladot")

#Evaluating model on test data(unseen data)
fit_svm_bal <- predict(svm_model_bal, test_breast_cancer)

#Table to calculate the Accuracy, precision, recall and F-scores
tab_bal <- table(fit_svm_bal, test_breast_cancer$Status)

svm_cm_bal <- confusionMatrix(tab_bal, positive = "0")
svm_cm_bal

#Results
print(paste("For Imbalanced data:", "Precision is:",caret::precision(tab_bal),
"Recall is:",sensitivity(tab_bal),"F-score is:",caret::F_meas(tab_bal)))

```




#Model2(Decision trees)#Imbalanced data(Model Training)
```{r}
set.seed(123)
#imbalanced data
dt_imb <- C5.0(train_breast_cancer[,-27], train_breast_cancer$Status)

fit_dt_imb <- predict(dt_imb, test_breast_cancer)

dt_imb_tab <- table(fit_dt_imb, test_breast_cancer$Status)

dt_cm_imb <- confusionMatrix(dt_imb_tab)

dt_cm_imb

CrossTable(test_breast_cancer$Status, fit_dt_imb,
 prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
 dnn = c('actual default', 'predicted default'))

#Results
print(paste("For Imbalanced data:", "Precision is:",caret::precision(dt_imb_tab),
"Recall is:",sensitivity(dt_imb_tab),"F-score is:",caret::F_meas(dt_imb_tab)))
```



#Model2(Decision trees)#Balanced data(Model Training)
```{r}
set.seed(123)
#imbalanced data
dt_bal <- C5.0(train_breast_cancer_smote[,-27], train_breast_cancer_smote$Status)

fit_dt_bal <- predict(dt_bal, test_breast_cancer)

dt_bal_tab <- table(fit_dt_bal, test_breast_cancer$Status)

dt_cm_bal <- confusionMatrix(dt_bal_tab, positive = "0")

dt_cm_bal

CrossTable(test_breast_cancer$Status, fit_dt_bal,
 prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
 dnn = c('actual default', 'predicted default'))

#Results
print(paste("For Imbalanced data:", "Precision is:",caret::precision(dt_bal_tab),
"Recall is:",sensitivity(dt_bal_tab),"F-score is:",caret::F_meas(dt_bal_tab)))
```

#Model3(Logistic Regression)#Imbalanced data(model training)
```{r}
#Performing logistic regression model on imbalanced data
log_imb <- glm(formula = Status ~ ., family = binomial(link = "logit"), 
    data = train_breast_cancer)

fitted.results <- predict(log_imb,newdata=test_breast_cancer,type='response')
fitted.results <- ifelse(fitted.results > 0.5,1,0)
tab_fitted <- table(fitted.results, test_breast_cancer$Status)

misClasificError <- mean(fitted.results != test_breast_cancer$Status)
print(paste('Accuracy',1-misClasificError))



#Backward Elimination
Backward_log_imb <- step(log_imb, direction = "backward", trace = TRUE)
summary(Backward_log_imb)


fitted.results1 <- predict(Backward_log_imb,newdata=test_breast_cancer,type='response')
fitted.results1 <- ifelse(fitted.results1 > 0.5,1,0)
tab_fitted1 <- table(fitted.results1, test_breast_cancer$Status)

misClasificError <- mean(fitted.results != test_breast_cancer$Status)
print(paste('Accuracy',1-misClasificError))

misClasificError1 <- mean(fitted.results1 != test_breast_cancer$Status)
print(paste('Accuracy',1-misClasificError1))

confusionMatrix(table(fitted.results, test_breast_cancer$Status))
log_cm_imb <- confusionMatrix(table(fitted.results1, test_breast_cancer$Status))


#Results
#Before backward elimination
print(paste("For Imbalanced data:", "Precision is:",caret::precision(tab_fitted),
"Recall is:",sensitivity(tab_fitted),"F-score is:",caret::F_meas(tab_fitted)))
#After Backward elimination
print(paste("For Imbalanced data", "after backward elimination:", "Precision is:",caret::precision(tab_fitted1),
"Recall is:",sensitivity(tab_fitted1),"F-score is:",caret::F_meas(tab_fitted1)))
#We see that, a little improvement in the values after backward elimination
#step() could not remove all the non-signficant variables in the model. we 
#can manually drop the non-significant variables having p-value of above 0.05
```



##Model3(Logistic Regression)#Balanced data(model training)
```{r}
#Performing logistic regression model on Balanced data
log_bal <- glm(formula = Status ~ ., family = binomial(link = "logit"), 
    data = train_breast_cancer_smote)


fitted.results_bal <- predict(log_bal,newdata=test_breast_cancer,type='response')
fitted.results_bal <- ifelse(fitted.results_bal > 0.5,1,0)
tab_fitted_bal <- table(fitted.results, test_breast_cancer$Status)

misClasificError <- mean(fitted.results_bal != test_breast_cancer$Status)
print(paste('Accuracy',1-misClasificError))




#Backward Elimination
Backward_log_bal <- step(log_bal, direction = "backward", trace = TRUE)
summary(Backward_log_bal)


fitted.results1_bal <- predict(Backward_log_bal,newdata=test_breast_cancer,type='response')
fitted.results1_bal <- ifelse(fitted.results1_bal > 0.5,1,0)
tab_fitted1_bal <- table(fitted.results1_bal, test_breast_cancer$Status)

misClasificError <- mean(fitted.results1_bal != test_breast_cancer$Status)
print(paste('Accuracy',1-misClasificError))

misClasificError1 <- mean(fitted.results1 != test_breast_cancer$Status)
print(paste('Accuracy',1-misClasificError1))

confusionMatrix(table(fitted.results_bal, test_breast_cancer$Status))
log_cm_bal <- confusionMatrix(table(fitted.results1_bal, test_breast_cancer$Status))


#Results
#Before backward elimination
print(paste("For Balanced data:", "Precision is:",caret::precision(tab_fitted_bal),
"Recall is:",sensitivity(tab_fitted_bal),"F-score is:",caret::F_meas(tab_fitted_bal)))
#After Backward elimination
print(paste("For Balanced data","after backward elimination:", "Precision is:",caret::precision(tab_fitted1_bal),
"Recall is:",sensitivity(tab_fitted1_bal),"F-score is:",caret::F_meas(tab_fitted1_bal)))
#We see that, a little improvement in the values after backward elimination
#step() could not remove all the non-signficant variables in the model. we 
#can manually drop the non-significant variables having p-value of above 0.05
#We can evaluate the model performance using the both holdout method and 
#k-fold cross validation method
```




#Model Evaluation(Holdout method)(support vector machines)
```{r}
#setting seed to produce same results each time
set.seed(123)
#Performing model evaluation using holdout method the imbalanced training data 
#on support vector machines
#model
#The trainControl() function is used to create a set of configuration options 
#known as a control object. This object guides the train() function and allows 
#for the selection of model evaluation criteria, such as the resampling strategy 
#and the measure used for choosing the best model
Grid_svm <- expand.grid(C=c(1:10))
ctrl <- trainControl(method = "LGOCV", p=0.75)
model_svm <- train(Status ~ ., data = train_breast_cancer, method = "svmLinear",
                   trControl = ctrl, tuneGrid=Grid_svm)
model_svm

fit_svm <- predict(model_svm, test_breast_cancer)

table(fit_svm, test_breast_cancer$Status)


##Performing model evaluation using holdout method of the balanced training data
#on support vector machines
#model
ctrl1 <- trainControl(method = "LGOCV", p=0.75)
model_svm_bal <- train(Status ~ ., data = train_breast_cancer_smote, 
                       method = "svmLinear", trControl = ctrl1, tuneGrid=Grid_svm)
model_svm_bal

fit_svm1 <- predict(model_svm_bal, test_breast_cancer)

table(fit_svm1, test_breast_cancer$Status)
```
Results: For support vector machines, Holdout method output showing that, kappa 
statistic of the balanced data would be 0.606 at tuning parameter C value of 3 
and for imbalanced data it would be 0.48 at C parameter value 6, In the next 
steps, we will tune the hyper parameter to check the any difference in the 
outputs and will consider the optimal model.

From the output, Tuning C hyper parameter of SVM for holdout method produces 
fairly similar results for both imbalanced and balanced data. C=6 is best 
parameter for imbalanced data and C=3 best for balanced data.



#Model Evaluation(Holdout Method)(Decision tree)
```{r,warning=FALSE}
#setting seed to get reproducible results
set.seed(123)
#Performing model evaluation using holdout method by taking 
#partition of 0.75 of the imbalanced training data on decision tree algorithm
#using C5.0 
Grid <- expand.grid(model="tree", trials=c(1,5,10,15,20,25,30),winnow=FALSE)

ctrl <- trainControl(method = "LGOCV", p=0.8, selectionFunction = "oneSE")
model_DT <- train(Status ~ ., data = train_breast_cancer, method = "C5.0",
                   trControl = ctrl, tuneGrid=Grid)
model_DT

fit_DT <- predict(model_DT, test_breast_cancer)

table(fit_DT, test_breast_cancer$Status)


##Performing model evaluation using holdout method of the balanced training data
#on decision tree algorithm
#model
ctrl <- trainControl(method = "LGOCV", p=0.8, selectionFunction = "oneSE")
model_DT_bal <- train(Status ~ ., data = train_breast_cancer_smote, 
                       method = "C5.0", trControl = ctrl, tuneGrid=Grid)
model_DT_bal

fit_DT1 <- predict(model_DT_bal, test_breast_cancer)

table(fit_DT1, test_breast_cancer$Status)

```
Results: Holdout method output showing that, kappa statistic of the balanced
data would be optimal at trials=30 with value of 0.78 and for imbalanced 
data optimal model was at trials=1 wit the value of 0.52.In the next 
steps, we will tune the hyperparameter to check the any difference in the 
outputs and will consider the optimal model. In the next steps, no need to
tune the hyperparameters.we already did it here with different trials values.




#Model Evaluation(K-fold cross validation)(support vector machines)
```{r}
#setting seed to produce same results each time
set.seed(123)
#Performing model evaluation using k-fold cross validation by taking 10
#repeated folds of the imbalanced training data on support vector machines
#model
#The trainControl() function is used to create a set of configuration options 
#known as a control object. This object guides the train() function and allows 
#for the selection of model evaluation criteria, such as the resampling strategy 
#and the measure used for choosing the best model
Grid_svm <- expand.grid(C=c(1,2,3,4,5,6,7,8,9,10))
ctrl <- trainControl(method = "cv", number = 10)
model_svm <- train(Status ~ ., data = train_breast_cancer, method = "svmLinear",
                   trControl = ctrl, tuneGrid=Grid_svm)
model_svm

fit_svm <- predict(model_svm, test_breast_cancer)

table(fit_svm, test_breast_cancer$Status)


##Performing model evaluation using k-fold cross validation by taking 10
#repeated folds of the balanced training data on support vector machines
#model
ctrl1 <- trainControl(method = "cv", number = 10)
model_svm_bal <- train(Status ~ ., data = train_breast_cancer_smote, 
                       method = "svmLinear", trControl = ctrl1, tuneGrid=Grid_svm)
model_svm_bal

fit_svm1 <- predict(model_svm_bal, test_breast_cancer)

table(fit_svm1, test_breast_cancer$Status)
```
Results: The kappa is about "0.59" for balanced data, which agrees with the previous
confusion matrix() from caret (the small difference is due to rounding). 
Using the suggested interpretation, we note that there is good agreement between
the classifier's predictions and the actual values on different validation
sets.

The final kappa was "0.48" for imbalanced data which is really moderate. It 
indicates that the model is no better at predicting then chance alone.

Imbalanced data got more accuracy because this is especially important for 
data sets with severe class imbalance because a classifier can obtain high 
accuracy simply by always guessing the most frequent class. The kappa statistic
will only reward the classifier if it is correct more often than this 
simplistic strategy.

Tuning C hyper parameter in this model was of no use. Every "C" parameter produces
exactly same results. So, default value of C is better to consider.




#Model Evaluation(K-fold cross validation)(Decision Trees)
```{r,warning=FALSE}
#setting seed to get reproducible results
set.seed(123)
#Performing model evaluation using k-fold cross validation by taking 10
#repeated folds of the imbalanced training data on decision tree algorithm
#using C5.0 
Grid <- expand.grid(model="tree", trials=c(1,5,10,15,20,25,30),winnow=FALSE)

ctrl <- trainControl(method = "cv", number = 10, selectionFunction = "oneSE")
model_DT <- train(Status ~ ., data = train_breast_cancer, method = "C5.0",
                   trControl = ctrl,tuneGrid=Grid)
model_DT

fit_DT <- predict(model_DT, test_breast_cancer)

confusionMatrix(table(fit_DT, test_breast_cancer$Status))


##Performing model evaluation using k-fold cross validation by taking 10
#repeated folds of the balanced training data on decision trees
#model
set.seed(123)
ctrl <- trainControl(method = "cv", number = 10, selectionFunction = "oneSE")
model_DT_bal <- train(Status ~ ., data = train_breast_cancer_smote, 
                       method = "C5.0", trControl = ctrl,tuneGrid=Grid)
model_DT_bal

fit_DT1 <- predict(model_DT_bal, test_breast_cancer)

confusionMatrix(table(fit_DT1, test_breast_cancer$Status))
```
Results:The best model here is with balanced data having kappa value of 0.805
at trials = 30 and it is comparatively greater than the kappa value for 
imbalanced data of trials =1, because here we used selectionfunction "oneSE" 
instead of base function to get the optimal model.

For balanced data, it produces optimal model with trials=30
For imbalanced data, it produces optimal model with trials=1

Finally, by comparing two decision tree k-fold cross validation, I would
probably choose balanced data with trials=30. In the next steps, no need to
tune the hyper parameters.we already did it here with different trials values.



#Model Evaluation(Holdout method)(Logistic regression)
```{r,warning=FALSE}
#setting seed to produce same results each time
set.seed(123)
#Performing model evaluation using logistic regression of the imbalanced training 
#data

#The trainControl() function is used to create a set of configuration options 
#known as a control object. This object guides the train() function and allows 
#for the selection of model evaluation criteria, such as the resampling strategy 
#and the measure used for choosing the best model

ctrl <- trainControl(method = "LGOCV", p=0.8)
model_log <- train(Status ~ ., data = train_breast_cancer, method = "glm",
                   family=binomial(link="logit"),trControl = ctrl)
model_log

fit_log <- predict(model_log, test_breast_cancer)

table(fit_log, test_breast_cancer$Status)

set.seed(123)
##Performing model evaluation using holdout method of partition 0.75
#of the balanced training data on logistic regression model
ctrl1 <- trainControl(method = "LGOCV", p=0.8)
model_log_bal <- train(Status ~ ., data = train_breast_cancer_smote, 
                       method = "glm",family=binomial(link="logit"),
                       trControl = ctrl1)
model_log_bal

fit_log1 <- predict(model_log_bal, test_breast_cancer)

table(fit_log1, test_breast_cancer$Status)

```



#Model Evaluation(k-fold cross validation method)(Logistic regression)
```{r,warning=FALSE}
#setting seed to get reproducible results
set.seed(123)
#Performing model evaluation using k-fold cross validation by taking 10
#repeated folds of the imbalanced training data on logistic regression algorithm
#using glm with parameter binomial(link="logit")

ctrl <- trainControl(method = "cv", number = 10, selectionFunction = "oneSE")
model_log_k <- train(Status ~ ., data = train_breast_cancer, method = "glm",
                   family=binomial(link="logit"),trControl = ctrl)
model_log_k

fit_log_k <- predict(model_log_k, test_breast_cancer)

table(fit_log_k, test_breast_cancer$Status)


set.seed(123)
##Performing model evaluation using k-fold cross validation by taking 10
#repeated folds of the balanced training data on logistic regression
#model
ctrl <- trainControl(method = "cv", number = 10, selectionFunction = "oneSE")
model_log_bal_k <- train(Status ~ ., data = train_breast_cancer_smote, 
                       method = "glm", family=binomial(link="logit"),trControl = ctrl)
model_log_bal_k

fit_log_bal_k <- predict(model_log_bal_k, test_breast_cancer)

table(fit_log_bal_k, test_breast_cancer$Status)
```
Results: For support vector machines, Holdout method output showing that, kappa 
statistic of the balanced data would be 0.606 at tuning parameter C value of 3 
and for imbalanced data it would be 0.48 at C parameter value 6, In the next 
steps, we will tune the hyper parameter to check the any difference in the 
outputs and will consider the optimal model.

From the output, Tuning C hyper parameter of SVM for holdout method produces 
fairly similar results for both imbalanced and balanced data. C=6 is best 
parameter for imbalanced data and C=3 best for balanced data.

Results: The kappa is about "0.59" for balanced data, which agrees with the previous
confusion matirx() from caret (the small difference is due to rounding). 
Using the suggested interpretation, we note that there is good agreement between
the classifier's predictions and the actual values on different validation
sets.

The final kappa was "0.48" for imbalanced data which is really moderate. It 
indicates that the model is no better at predicting then chance alone.

Imbalanced data got more accuracy because this is especially important for 
data sets with severe class imbalance because a classifier can obtain high 
accuracy simply by always guessing the most frequent class. The kappa statistic
will only reward the classifier if it is correct more often than this 
simplistic strategy.

Tuning C hyper parameter in this model was of no use. Every "C" parameter produces
exactly same results. So, default value of C is better to consider.

Results:The best model here is with balanced data having kappa value of 0.805
at trials = 20 and it is comparatively greater than the kappa value for 
imbalanced data of trials =1, because here we used selectionfunction "oneSE" 
instead of base function to get the optimal model.

For balanced data, it produces optimal model with trials=20
For imbalanced data, it produces optimal model with trials=1

Finally, by comparing two decision tree k-fold cross validation, I would
probably choose balanced data with trials=20. In the next steps, no need to
tune the hyper parameters.we already did it here with different trials values.

Results:The best model here is with balanced data having kappa value of 0.805
at trials = 20 and it is comparatively greater than the kappa value for 
imbalanced data of trials =1, because here we used selectionfunction "oneSE" 
instead of base function to get the optimal model.

For balanced data, it produces optimal model with trials=20
For imbalanced data, it produces optimal model with trials=1

Finally, by comparing two decision tree k-fold cross validation, I would
probably choose balanced data with trials=20. In the next steps, no need to
tune the hyper parameters.we already did it here with different trials values.

Logistic regression of k-fold cross validation showing high number of false
negative and moderate agreement of actual and predicted values.

Comparing all the models: By comparing all the models above, I would probably 
choose decision tree on balanced data with kappa value of 0.8 and having low
false negatives and best predicting in negative class.


#Model4(Random Forests k-fold)#Imbalanced data
```{r,warning=FALSE,message=FALSE}
library(randomForest)
#setting seed to random number to get reproducible results
set.seed(123)

ctrl <- trainControl(method = "cv", number = 10, selectionFunction = "oneSE")

rf_model <- train(Status ~., data=train_breast_cancer, method="rf", 
                  metric=c("Accuracy"), trControl=ctrl)

predict_rf_imb = predict(rf_model, newdata = test_breast_cancer)

# Confusion matrix on test set
confusionMatrix(table(predict_rf_imb, test_breast_cancer$Status))
```



#Model5(Random Forests)#Balanced data
```{r}
set.seed(100)

ctrl <- trainControl(method = "cv", number = 10, selectionFunction = "oneSE")

rf_model_bal <- train(Status ~., data=train_breast_cancer_smote, method="rf", 
                      metric=c("Accuracy"), trControl=ctrl)

predict_rf_bal = predict(rf_model_bal, newdata = test_breast_cancer)

# Confusion matrix on test set
confusionMatrix(table(predict_rf_bal, test_breast_cancer$Status))
```




#Model Tuning & performance improvement(Meta Learning)
#Bagging(With homogeneous learners)(same algorithms)
```{r,warning=FALSE}

RNGversion("3.5.2")
set.seed(300)

#Using bagging function to ensemble the model on imbalanced train data by
#taking nbags parameter as 25
bag_imb <- bagging(Status~., data=train_breast_cancer, nbag=25)

bag_pred_imb <- predict(bag_imb, test_breast_cancer)

tab_bag_imb <- table(bag_pred_imb, test_breast_cancer$Status)

confusionMatrix(tab_bag_imb)

RNGversion("3.5.2")
set.seed(300)
ctrl <- trainControl(method = "repeatedcv", number = 10)


train_bag <- train_breast_cancer
train_bag[,c(2:20,22,23)] <- lapply(train_bag[,c(2:20,22,23)], factor)

train(Status ~ ., data = train_bag[,c(1,21,24:27)], method="treebag",
 trControl = ctrl) #kappa : 0.49


#Using bagging function to ensemble the model on Balanced train data by
#taking nbags parameter as 25
bag_bal <- bagging(Status~., data=train_breast_cancer_smote, nbag=25)

bag_pred_bal <- predict(bag_bal, test_breast_cancer)

tab_bag_bal <- table(bag_pred_bal, test_breast_cancer$Status)

confusionMatrix(tab_bag_bal)

RNGversion("3.5.2")
set.seed(123)
ctrl <- trainControl(method = "cv", number = 10)
bag_cv_bal <- train(Status ~ ., data = train_breast_cancer_smote[,c(1,21,24:27)],
                    method = "treebag",trControl = ctrl)

bag_cv_bal   #kappa score: 0.769

```

Output: For Imbalanced data(Bagging)

The best C5.0 decision tree we adapted previously in this chapter had a 0.43 
kappa statistic, and the 0.76 kappa value for this model shows that the bagged 
tree model works very well. This demonstrates the effectiveness of 
ensemble methods: when working together, a group of simple learners can 
outperform extremely complex models.

For Imbalanced data(Bagging)
There is little change in the both accuracy and kappa statistic values.

Here, I found interesting that for the bagging process, the number of true
negatives(TN) increased for both imbalanced and balanced data sets from the 
previous decision tree algorithm 


#Construction of ensemble model as function
```{r}
Ensemble_model <- function(data, algorithms) {
  
  # Set the seed for reproducibility
  set.seed(100)
  
  # Set up the train control object for repeated cross-validation
  control_stacking <- trainControl(
    method = "cv", 
    number=10, 
    selectionFunction = "oneSE"
  )
  
  # Train the stacked models using caretList
  stacked_models <- caretList(
    Status ~ ., 
    data = data, 
    trControl = control_stacking, 
    methodList = algorithms,
    family=binomial(link="logit")
  )
  
  # Generate resampling results for the stacked models
  stacking_results <- resamples(stacked_models)
  
  # Return the summary of the resampling results
  return(stacking_results)
  
}

```



#Application of ensemble to make prediction
```{r}
algorithms <- c("svmLinear","C5.0","glm","treebag","rf")
#Imbalanced data
Ensemble_imb <- Ensemble_model(train_breast_cancer[,c(1,21,24:27)], algorithms)
#Balanced data
Ensemble_bal <- Ensemble_model(train_breast_cancer_smote[,c(1,21,24:27)], algorithms)

summary(Ensemble_imb)
summary(Ensemble_bal)

#Draw box plots to compare the models
scales <- list(x=list(relation="free"), y=list(relation="free"))
#Imbalanced data plots
bwplot(Ensemble_imb, scales=scales)

#Balanced data plots
bwplot(Ensemble_bal, scales=scales)

```
Results: From the results, None of the above methods performs well. But,
when compared to other decision tree and random forests were the best.The 
Random Forest, bagging are scored well in terms of precision and recall scores
when compared to other models and the oversampling strategy for the SMOTE 
approach after training and testing the models. The Support Vector Classifier 
and Logistic Regression models, which had the highest recall scores, did not do
as well in accurately predicting the 'Alive' class, though. This is due to the 
fact that for these models, the 'False Positive' value—which indicates the number
of times the model erroneously predicted that a patient was dead—was 
exceptionally high.
Using  distinct method SMOTE, we also attempted testing the models on datasets 
that were both imbalanced and balanced. The outcomes, however, fell short of 
expectations. One possible explanation for this is the lack of a significant 
correlation between the features in the data set and the target label. We need 
more variables to get accurate predictions. The performance metrics also 
demonstrated that these oversampling strategies did not always result in 
better forecasts.

#Majority voting(for binary classification)
```{r,warning=FALSE}
predict_status <- function(testdata){
  svm_pred <- predict(svm_model_bal, test_breast_cancer[,-27]) 
  log_reg_pred <- predict(log_bal, test_breast_cancer[,-27]) 
  dec_tree_pred <- predict(dt_bal, test_breast_cancer[,-27], type = "class")
  bag_pred <- predict(bag_bal, test_breast_cancer[,-27])
 pred_majority <-  ifelse(sum(svm_pred == "1") + sum(log_reg_pred== "1") + sum(dec_tree_pred == "1") + sum(bag_pred==1) > 1, "1", "0")
 
 return(pred_majority)
}


predict_status(test_breast_cancer)
```
Ensemble method was working well on the test data by majority voting




#Feature engineering task(Additional support)
```{r}
#cancer_basic <- SEER_breast_cancer_df



#cancer_basic[,c(9,12:14)] <- lapply(cancer_basic[,c(9,12:14)], normalize)


#hist(cancer_basic$Age)
#hist(cancer_basic$Tumor_size)
#hist(cancer_basic$Regional_nodes_examined)
#hist(cancer_basic$Regional_nodes_positive)
#hist(cancer_basic$Survival_months)


#skewness(cancer_basic$Tumor_size)
#skewness(cancer_basic$Regional_nodes_examined)
#skewness(cancer_basic$Regional_nodes_positive)
#skewness(cancer_basic$Survival_months)

#skewness((cancer_basic$Tumor_size))
#skewness(sqrt(cancer_basic$Regional_nodes_examined))
#hist((sqrt(cancer_basic$Regional_nodes_positive)))
#skewness((log10(cancer_basic$Survival_months)))

```



#CRISP-DM APPROACH:

For my machine learning project, I have chosen the SEER Breast cancer Dataset. 
This data set contains information on over 4024 patients and includes features 
such as age, marital status, race, tumor_size, survival months and whether or 
not the patient alive or dead. The data set can be found on ieee website at the 
following URL: https://ieee-dataport.org/open-access/seer-breast-cancer-data
Links to an external site.

My goal is to develop a predictive model to identify individuals alive or dead 
based on their Age, marital status, race and other characteristics. The target
variable is binary variable(alive or dead). So, this is classification task

The target variable is one of 15 features (variables) in the data set, which 
includes 4024 rows overall.Both category and numerical features are present, 
and some of them have missing values that call for imputation or elimination. 
The use of machine learning and statistical methods to derive insights from the 
data and anticipate future outcomes makes this task suitable for classification 
as a data mining task.


I intend to test a variety of algorithms in order to construct the predictive 
model and determine the one that works best in this situation. I'll be using 
random forests, decision trees, support vector machines, and logistic regression 
among other approaches.These algorithms were chosen because they can handle 
both numerical and categorical data and are effective for binary classification 
problems.

BUSINESS UNDERSTANDING:
Breast cancer prediction using machine learning entails analyzing data related 
to breast cancer, such as patient demographics, medical history, genetic factors, 
and imaging results, using algorithms and statistical models. The goal is to 
create accurate models that can predict a patient's risk of getting breast cancer
or the chance of recurrence in individuals who have already been diagnosed. 
Health care providers can use these models to identify patients who are at high
risk for breast cancer and suggest screening and preventative procedures. They 
can also be utilized to create personalized treatment strategies for patients 
based on their unique risk factors.

Companies that sell breast cancer preventative measures and treatment products 
and services can utilize machine learning algorithms to identify potential clients 
and customize their advertising messages. A company that sells bras for breast 
cancer survivors, for example, can use machine learning to identify people who 
have had mastectomies and target them with personalized advertisements. Machine 
learning algorithms can be used to examine a massive amount of data and identify
patterns and trends that human researchers may not notice. This can assist 
companies develop new breast cancer treatments and therapies, as well as find new risk 
factors and prevention strategies.

Furthermore, the use of machine learning to predict breast cancer has the 
potential to increase the efficiency and accuracy of breast cancer diagnosis. 
Machine learning algorithms, for example, can be used to scan mammograms and 
indicate concerning areas that may require further evaluation. This can help 
reduce the number of false positives and false negatives, boosting overall breast 
cancer screening accuracy.



References:
Rabiei, R., Ayyoubzadeh, S. M., Sohrabei, S., Esmaeili, M., & Atashi, A. (2022, June 1). Prediction of breast cancer using machine learning approaches. Journal of biomedical physics & engineering. Retrieved April 26, 2023, from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9175124/#:~:text=The%20proposed%20machine%2Dlearning%20approaches,interventions%20at%20the%20right%20time.

Nasser, M., & Yusof, U. K. (2023, January 3). Deep learning based methods for breast cancer diagnosis: A systematic review and future direction. Diagnostics (Basel, Switzerland). Retrieved April 26, 2023, from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9818155/
Alzu’bi, A., Najadat, H., Doulat, W., Al-Shari, O., & Zhou, L. (2021, January 18).

Predicting the recurrence of breast cancer using machine learning algorithms - multimedia tools and applications. SpringerLink. Retrieved April 26, 2023, from https://link.springer.com/article/10.1007/s11042-020-10448-w#:~:text=Machine%20learning%20algorithms%20help%20physicians,and%20molecular%20subtype%20%5B35%5D%20.

