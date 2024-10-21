# Data Exploration (EDA)

# Encoding the target feature Status (Alive=0, Dead=1)
SEER_data$Status <- as.character(SEER_data$Status)
SEER_data$Status[SEER_data$Status == "Alive"] <- 0
SEER_data$Status[SEER_data$Status == "Dead"] <- 1
SEER_data$Status <- as.factor(SEER_data$Status)

# Factor analysis using bar plots of categorical variables to target variable to understand how different levels in the categorical variables respond to the target variable
cat_variables <- list("Race", "Marital_Status", "T_stage", "N_stage", "sixth_stage", "Grade", "A_stage", "Estrogen_status", "Progesterone_status")

# Created a loop to display all the bar plots at once in relation to target variable
par(mfrow = c(3, 3))
for (i in cat_variables) {
  gg_plot <- ggplot(SEER_data, aes_string(x = i, fill = SEER_data$Status)) +
    geom_bar(stat = "count") +
    scale_fill_discrete(name = "Status") +
    geom_text(aes(label = paste(after_stat(round(count / sum(count) * 100, 1)), "%")), stat = 'count', nudge_y = 0.125)
  print(gg_plot)
}

# To determine the relationship between continuous variables to target variable
par(mfrow = c(1, 1))
byf.hist(Age ~ Status, data = SEER_data)
