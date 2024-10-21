# SEER Breast Cancer Prediction Project
This project aims to develop a predictive model for identifying breast cancer patients' survival outcomes based on various features such as age, marital status, race, tumor size, survival months, and other medical attributes. The project uses multiple machine learning algorithms to classify whether a patient is alive or deceased. Key features include data preprocessing, exploratory data analysis (EDA), feature engineering, and model building with several classification algorithms.

## Table of Contents
- [About the Project](#about-the-project)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [License](#license)
- [Contact](#contact)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)

## About the Project
This project uses the SEER Breast Cancer Dataset, which contains information on over 4,024 patients. The goal is to develop a machine learning model that can predict whether a patient is alive or deceased based on their medical data. Various machine learning algorithms such as Support Vector Machines (SVM), Decision Trees, Logistic Regression, and Random Forests are used, and their performance is evaluated using different metrics.

Key features:
- Data Acquisition and Cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Model Building and Evaluation (SVM, Decision Trees, Logistic Regression, Random Forests)
- Ensemble Learning with Bagging and Majority Voting

## Getting Started
### Prerequisites
To run this project, you'll need the following software and dependencies:

- R programming environment
- Required R packages: `corrplot`, `magrittr`, `dplyr`, `ggcorrplot`, `psych`, `RVAideMemoire`, `moments`, `tidyverse`, `CatEncoders`, `DMwR`, `kernlab`, `C50`, `gmodels`, `caret`, `Metrics`, `irr`, `plotly`, `cvms`, `ipred`, `caretEnsemble`, `randomForest`

Install the required R packages using the following command:

```r
# Example for installing dependencies
install.packages(c("corrplot", "magrittr", "dplyr", "ggcorrplot", "psych", "RVAideMemoire", "moments", "tidyverse", "CatEncoders", "DMwR", "kernlab", "C50", "gmodels", "caret", "Metrics", "irr", "plotly", "cvms", "ipred", "caretEnsemble", "randomForest"))
```

### Installation
Step-by-step guide to set up the project locally.

```bash
# Clone the repository
git clone https://github.com/sakhamreddy/MachineLearning-Signature-Project.git

# Navigate to the project directory
cd repository

# Install dependencies
Rscript -e 'install.packages(c("corrplot", "magrittr", "dplyr", "ggcorrplot", "psych", "RVAideMemoire", "moments", "tidyverse", "CatEncoders", "DMwR", "kernlab", "C50", "gmodels", "caret", "Metrics", "irr", "plotly", "cvms", "ipred", "caretEnsemble", "randomForest"))'
```

## Usage
To run the project, use the provided R scripts. The analysis involves the following steps:

### 1. Data Preprocessing
Data preprocessing includes importing the SEER dataset, handling missing values, removing duplicate columns, and transforming variables for better analysis. This step ensures that the data is clean and ready for exploratory analysis.

```r
# Example usage for data preprocessing
source("data_preprocessing.R")
```

### 2. Exploratory Data Analysis (EDA)
EDA is used to better understand the relationships between different features and the target variable. Techniques like histograms, bar plots, and correlation plots are used to visualize the data.

```r
# Example usage for exploratory data analysis
source("exploratory_data_analysis.R")
```

### 3. Feature Engineering
Feature engineering is used to create new features or transform existing features to improve model performance. This includes standardization, dummy encoding, and applying principal component analysis (PCA).

```r
# Example usage for feature engineering
source("feature_engineering.R")
```

### 4. Model Building and Evaluation
In this step, various machine learning models are built, trained, and evaluated. Models include Support Vector Machines, Decision Trees, Logistic Regression, Random Forests, and Ensemble Learning approaches.

```r
# Example usage for model building and evaluation
source("model_building.R")
```

### 5. Ensemble Learning with Bagging and Majority Voting
In this step, bagging, boosting, and majority voting ensemble techniques are applied to improve model accuracy.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
**Manikantareddy Sakhamreddy**  
Email: [manikantasakham09@gmail.com](mailto:manikantasakham09@gmail.com)  
GitHub: [sakhamreddy](https://github.com/sakhamreddy)

## Contributing
Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Acknowledgments
- Special thanks to the SEER Breast Cancer Dataset contributors and IEEE DataPort.
- References:
  - Rabiei R;Ayyoubzadeh SM;Sohrabei S;Esmaeili M;Atashi A; (n.d.). Prediction of breast cancer using machine learning approaches. Journal of biomedical physics & engineering. https://pubmed.ncbi.nlm.nih.gov/35698545/ 
  - Nasser, M., & Yusof, U. K. (2023, January 3). Deep learning based methods for breast cancer diagnosis: A systematic review and future direction. MDPI. https://www.mdpi.com/2075-4418/13/1/161 
  - Alzu&rsquo;bi, A., Najadat, H., Doulat, W., Al-Shari, O., & Zhou, L. (2021, January 18). Predicting the recurrence of breast cancer using machine learning algorithms - multimedia tools and applications. SpringerLink. https://link.springer.com/article/10.1007/s11042-020-10448-w
  
