# Loan Default Risk Analysis

## Overview
This repository contains an Exploratory Data Analysis (EDA) on a loan default risk dataset. The goal of this project is to analyze factors affecting loan repayment and default, perform feature engineering, and derive key insights that can help financial institutions make informed lending decisions.

## Dataset
The dataset consists of two main files:
1. `application_data.csv` - Contains information about loan applicants and their financial profiles.
2. `previous_application.csv` - Contains details of past loan applications by the applicants.

## Objectives
- Perform univariate and bivariate analysis to identify trends and insights.
- Examine relationships between various financial, demographic, and credit-related attributes.
- Identify key factors influencing loan defaults.
- Handle missing values and outliers effectively.

## Analysis Steps
1. **Data Cleaning & Preprocessing:**
   - Remove columns with more than 40% missing values.
   - Handle null values using appropriate imputation techniques.
   - Convert necessary columns to suitable data types.
   
2. **Univariate Analysis:**
   - Distribution of loan amounts and credit types.
   - Analysis of loan purpose and applicant income types.
   - Contract approval status breakdown.
   
3. **Bivariate Analysis:**
   - Credit amount vs education type for on-time repayers vs defaulters.
   - Credit amount vs income type for on-time repayers vs defaulters.
   - Housing type vs credit amount.
   - Client type vs credit amount.
   
4. **Correlation Analysis:**
   - Compute correlation matrices for both repayers and defaulters.
   - Identify top correlated features affecting loan repayment.
   
5. **Outlier Detection & Handling:**
   - Identify and visualize outliers in key numerical columns.
   - Apply appropriate transformations or removal techniques.

## Key Insights & Conclusions
### Repayers (On-Time Loan Payers)
- Higher-income individuals (>6L) are less likely to default.
- Academic degree holders have a lower probability of defaulting.
- Female applicants and business owners are more reliable borrowers.
- Revolving loans have a higher repayment rate.

### Defaulters
- Male applicants have a higher tendency to default.
- Unemployed and individuals on maternity leave face higher default rates.
- Applicants living in rented apartments default more frequently.
- Loan applicants from certain industries (e.g., Industry Types 3, 13, and 8) have higher default risks.
- Loans for repairs have higher rejection rates due to higher risk.

## Technologies Used
- **Python**
- **Pandas** for data manipulation
- **Matplotlib & Seaborn** for data visualization
- **NumPy** for numerical computations
- **Scikit-learn** for feature analysis (if applicable)



