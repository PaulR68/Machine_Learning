# Multi-Source Socio-Economic Classification Project

## Project Overview
This project focuses on building a machine learning pipeline to predict a binary target variable ('T' vs. 'L') for individuals based on a complex, multi-source dataset. The data includes information regarding demographics, employment history, sports club memberships, and geographical metrics.

The primary objective was to consolidate these dispersed data sources into a coherent feature set and train a robust classifier capable of generalizing well on unseen test data.

## Key Challenge
The main difficulty of this project lay in the data engineering phase. The raw data was fragmented across multiple CSV files (main dataset, sports, employment, retirement, geography) with varying levels of granularity. Significant effort was required to:
1. Merge relational datasets while preventing data leakage.
2. Handle high-cardinality categorical variables (e.g., job descriptions, sports clubs).
3. Impute missing values using hierarchical strategies based on socio-economic logic rather than simple averages.

## Methodology

### 1. Data Preprocessing
The data cleaning process involved extensive feature engineering:
- **Sports Data:** Aggregated 97 different sports clubs into broader categories based on club size.
- **Geography:** Integrated external data including city density, median revenue, and population counts, while mapping specific city districts to global codes.
- **Employment:** Mapped complex job descriptions to N2 nomenclature and imputed missing working hours using a cascading grouping strategy (by CSP and contract type).

### 2. Model Selection and Optimization
We established a baseline using a Decision Tree (Accuracy: ~72%). We then compared several advanced algorithms, including Random Forest and XGBoost. The optimization process utilized GridSearchCV with Stratified K-Fold validation.

We specifically chose the **F1-Score on the minority class ('T')** as our optimization metric to address slight class imbalance and ensure the model did not simply favor the majority class.

## Results
**XGBoost** was selected as the final model due to its superior handling of missing values and heterogeneous features.

* **Validation Accuracy:** 75.94% (+3 points over Random Forest).
* **AUC Score:** 0.83, indicating strong separability between classes.
* **Weighted F1-Score:** 0.76.

Feature importance analysis revealed that the model correctly prioritized professional stability (Activity Status, Employment Tenure) over noisy geographical data, confirming the model learned valid socio-economic patterns rather than overfitting.

## Repository Structure
* `Part1.py`: Contains the `DataPreprocessor` class and the full ETL pipeline for cleaning and merging datasets.
* `Part2.ipynb`: Exploratory data analysis, model comparison, and hyperparameter tuning (GridSearchCV).
* `Part3.ipynb`: Loads the pre-trained pipeline and generates the final predictions (`predictions.csv`) on the test set.
* `Report_ML`: Showcase the final report, including a developed explanation of each step of the project
