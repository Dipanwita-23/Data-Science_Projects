
#  Bank Loan Analysis

This project involves a comprehensive analysis of bank loan data, aimed at understanding key performance metrics such as loan applications, funding amounts, and loan performance over time. The project includes an SQL-based data analysis and a descriptive dashboard created using Power BI to visualize insights.

![Alt text](https://github.com/Dipanwita-23/Data-Science_Projects/blob/main/Bank_Loan_Analysis_Using_SQL_PowerBi/Bank%20loan%20Report_Overview.png)

## Purpose
This Bank Loan Analysis project is essential for optimizing the bank’s loan portfolio, minimizing risks, and increasing profitability. The need for such a project arises from several key business objectives:

- Risk Management: By analyzing loan performance (good vs. bad loans), the bank can improve its loan approval criteria, reducing the risk of defaults and non-performing loans.

- Profitability Insights: Understanding trends in total loan applications, funded amounts, and repayments helps the bank maximize profit by identifying the most lucrative loan products and interest rate structures.

- Geographical Performance: Analyzing loan data across different regions (e.g., identifying loss-making states) helps the bank make informed decisions about where to focus lending efforts or tighten risk controls.

- Loan Term Optimization: Short-term and long-term loan performance comparison enables the bank to fine-tune its loan offerings for better profitability and customer satisfaction.

- Targeted Marketing and Customer Segmentation: This project helps the bank better understand customer profiles (loan purpose, employment status, etc.), allowing for more effective, tailored marketing strategies.

- Strategic Decision-Making: The insights from this analysis empower the bank to plan future strategies, allocate resources efficiently, and maintain a competitive edge in the financial market.
## Project Overview
The analysis focuses on evaluating loan applications and their performance based on multiple dimensions like:

- Total loan applications
- Funded amounts
- Loan repayment status
- Interest rates
- Debt-to-income (DTI) ratios
- Loan quality (good vs. bad loans)

Both monthly and previous-month-to-date (PMTD) metrics are included to track performance trends.
## Key Matrics
- Total Loan Applications: The total number of loan applications received.

- Funded Amount: The total amount of loans funded.

- Amount Received: The total repayments received from funded loans.

- Average Interest Rate: The mean interest rate for funded loans.

- Debt-to-Income Ratio (DTI): The average DTI for loan applicants.

- Good Loan Issued: Loans that meet the bank’s quality criteria.

- Bad Loan Issued: Loans that have a high likelihood of default or poor performance.
## Loan Status Analysis
The analysis is broken down by various factors, including:

- Month-wise performance: Tracking the change in loan performance over months.
- State-wise distribution: Analysis based on geographical data to - see where most loans are issued and their performance.
- Term Length: Loan performance based on short-term vs long-term loans.
- Employment Length: Correlation between loan performance and employment history.
- Purpose: Understanding how loan purposes affect their approval and performance.
- Home Ownership: Analyzing how owning a home impacts loan application success and repayment performance.

## Dashboard
The Power BI dashboard provides an interactive interface for exploring the data. It includes three dashboards **Summary**, **Overview** and **Details**. This visualizes:

- Trends in loan applications and funding amounts over time.
- Comparisons between Good Loans and Bad Loans.
- Detailed metrics such as Average Interest Rate and Debt-to-Income Ratio.
- State-wise and term-wise loan performance.
- Net profit/loss accross different purposes and states.
- The overall summary of different loans like Currnt, Fully Paid and Charged Off.
![Alt text](https://github.com/Dipanwita-23/Data-Science_Projects/blob/main/Bank_Loan_Analysis_Using_SQL_PowerBi/Bank%20loan%20Report_Summary.png)
![Alt text](https://github.com/Dipanwita-23/Data-Science_Projects/blob/main/Bank_Loan_Analysis_Using_SQL_PowerBi/Bank%20loan%20Report_Overview.png)
![Alt text](https://github.com/Dipanwita-23/Data-Science_Projects/blob/main/Bank_Loan_Analysis_Using_SQL_PowerBi/Bank%20loan%20Report_Details.png)

## Data Source
[Download the dataset]((https://github.com/Dipanwita-23/Data-Science_Projects/blob/main/Bank_Loan_Analysis_Using_SQL_PowerBi/financial_loan.csv)

## Tools Used

- SQL Queries: The SQL file included in the repository contains queries for generating all the key metrics and insights used in the analysis.
[Download PDF file](https://github.com/Dipanwita-23/Data-Science_Projects/blob/main/Bank_Loan_Analysis_Using_SQL_PowerBi/Bank%20loan%20analysis_SQL.pdf)
- Power BI Dashboard: The dashboard file can be opened in Power BI to interact with and explore the data.

[Download Power BI file](https://github.com/Dipanwita-23/Data-Science_Projects/blob/main/Bank_Loan_Analysis_Using_SQL_PowerBi/Bank%20loan%20analysis%20dashboard.pbix)

## Key Insights

- Increasing Loan Metrics: The total loan applications, funded amounts, and amounts received are showing an upward trend, indicating growing loan activity.
- Net Profit: The overall net profit from loans is ₹37,313,858.
- Loan Quality: Good Loans account for 86% of the total loans issued, while Bad Loans represent 13%, showing strong loan screening and risk management.
- Average Interest Rate: The average interest rate across all loans is 12%, contributing to the profitability of the loan portfolio.
- Short-Term vs Long-Term Loans: 73% of the loans are short-term, contributing to a profit of ₹21M. 27% of the loans are long-term, with a profit of ₹16M.
- Geographical Performance: States like Nebraska (NE), Tennessee (TN), and Indiana (IN) in the U.S. are incurring losses, highlighting regional risk that requires targeted interventions.

## Conclusion
This project provides a detailed analysis of bank loan data, allowing for a deeper understanding of loan performance across different dimensions. It helps in identifying key trends, potential risk areas, and opportunities for improvement in the loan approval and funding process.
