# Loan Approval Classification: Comprehensive Analysis

## Project Goal

This project executes a complete Machine Learning workflow, including advanced **Feature Engineering**, **Multiple Model Training**, and **SHAP Feature Importance Analysis**, to predict the binary target variable, `loan_status` (1=Approved, 0=Rejected). The objective is not only to achieve high predictive accuracy but also to thoroughly explain the key drivers behind the approval decision.

## Methodology and Technical Execution

### Technologies Used

* **Language:** Python
* **Core Libraries:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `shap` (for explainability)

### Modeling and Validation Strategy

The workflow employed involved comparing multiple classification techniques:
1.  **Baseline Model:** Logistic Regression (Linear)
2.  **Ensemble Models:** Random Forest and Gradient Boosting (Non-linear)

The **Random Forest** model was selected for deployment due to its superior F1-score performance (balancing Precision and Recall) and better interpretability through the SHAP framework.

## Analysis and Key Findings (SHAP Explainability)

The SHAP (SHapley Additive exPlanations) analysis provided crucial insights into the model's decision-making process:

### Positive Drivers for Approval

* **Creditworthiness:** Features such as high `Credit_Score` and low `Debt_to_Income` ratio are the strongest positive contributors to the probability of approval (positive SHAP values).
* **Loan Intent:** Specific loan intents, particularly 'Education' and 'Home Improvement', positively influence the decision.

### Negative Drivers for Approval

* **Financial Burden:** Features like high `interest_rate` and the engineered `Debt_Level` (specifically the 'High\_Debt' category) have the strongest negative impact on loan approval, indicating lenders heavily mitigate risk associated with high existing debt.
* **Loan Intent:** 'Debt Consolidation' and 'Business' loan intents carry higher perceived risk, negatively impacting approval probability.

## Project Setup

To clone and execute this analysis locally, ensure you have the required dependencies installed.

```bash
git clone [Your Repository URL Here]
cd Loan-Approval-Classification-ML
pip install -r requirements.txt
jupyter notebook Loan_Approval_Classification2.ipynb
