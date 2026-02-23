# Credit Risk Scoring Model
### McMillin Analytics

A machine learning model that predicts loan default probability using borrower fundamentals — no proprietary credit grades required. Features domain-driven feature engineering, a grade vs. no-grade model comparison, and an interactive Streamlit application that generates implied risk grades from borrower data.

**[Live Demo →](https://credit-risk-model-3pyy8acs9ypikqxzwcjh6g.streamlit.app/)**

---

## Overview

Most credit risk models treat default prediction as a pure data science exercise. This project takes a different approach — feature engineering, evaluation metrics, and business context are grounded in credit analysis fundamentals. The model doesn't just predict defaults; it explains *why* in terms a credit committee would understand.

A key finding: **removing Lending Club's proprietary grade — which accounted for 68% of feature importance — reduced AUC by only 0.0018.** The borrower fundamentals captured through feature engineering contain nearly the same risk signal as the platform's internal grading system. The production model uses no grade inputs, relying entirely on engineered features to predict default probability and generate an implied risk grade.

## Key Features

- **Classification models** (Logistic Regression, Random Forest, XGBoost) trained on 2.2M+ Lending Club loans
- **Domain-driven feature engineering** — features modeled on the 5 C's of Credit framework, including a pro forma DTI that incorporates the proposed loan payment
- **Grade vs. no-grade model comparison** — demonstrates that engineered borrower fundamentals capture nearly all the predictive power of the platform's proprietary grading
- **Implied risk grading** — the model predicts default probability and maps it to a letter grade, rather than relying on the platform's assigned grade as an input
- **Interactive Streamlit app** — input borrower characteristics, receive a real-time risk score with an auto-generated credit memo

## Results

### Model Comparison: With Grade vs. Without Grade

| Metric | With Grade | Without Grade | Delta |
|--------|-----------|--------------|-------|
| AUC-ROC | 0.7275 | 0.7257 | -0.0018 |
| Avg Precision | 0.3999 | 0.3984 | -0.0015 |

Lending Club's grade and sub-grade together accounted for 79% of feature importance in the original model — essentially, the model was learning that Lending Club's own risk assessment was the best predictor of default. Removing these features forced the model to rely on borrower fundamentals, with virtually no loss in predictive power.

### All Models (With Grade)

| Model | AUC-ROC | Avg Precision | Accuracy | F1 Score |
|-------|---------|---------------|----------|----------|
| Logistic Regression | 0.7114 | 0.3729 | 0.6606 | 0.4294 |
| Random Forest | 0.7170 | 0.3863 | 0.6665 | 0.4330 |
| XGBoost (with grade) | 0.7275 | 0.3999 | 0.6591 | 0.4415 |
| **XGBoost (no grade — production)** | **0.7257** | **0.3984** | — | — |

### Top Features (No-Grade Model)

| Rank | Feature | Importance | Credit Rationale |
|------|---------|-----------|-----------------|
| 1 | Interest Rate | 25.0% | Reflects both pricing risk and payment burden |
| 2 | Loan Term | 11.7% | Longer tenor = longer exposure to default risk |
| 3 | Renter Status | 5.1% | Home ownership as a stability proxy |
| 4 | Pro Forma DTI | 3.5% | Post-closing debt burden — engineered feature |
| 5 | Loan-to-Income | 3.1% | Leverage relative to earnings |

## Technical Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| pandas / NumPy | Data manipulation |
| scikit-learn | Logistic Regression, Random Forest, preprocessing |
| XGBoost | Gradient boosted classification |
| matplotlib / seaborn | Statistical visualization |
| Plotly | Interactive charts |
| Streamlit | Web application |

## Project Structure

```
credit-risk-model/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_data_exploration.ipynb    # EDA with credit analyst lens
│   ├── 02_feature_engineering.ipynb # Domain-driven feature creation
│   └── 03_modeling.ipynb            # Model training, comparison & business impact
├── app/
│   └── app.py                       # Streamlit application
├── src/
│   └── data_cleaning.py
├── data/
│   ├── raw/                         # Original Lending Club data (not tracked)
│   └── processed/                   # Cleaned & engineered datasets
└── results/                         # Visualizations and model outputs
```

## Quick Start

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/credit-risk-model.git
cd credit-risk-model

# Install dependencies
pip install -r requirements.txt

# Download data from Kaggle:
# https://www.kaggle.com/datasets/wordsforthewise/lending-club
# Place accepted_2007_to_2018Q4.csv in data/raw/

# Run notebooks in order
jupyter notebook notebooks/01_data_exploration.ipynb

# Launch the app (after running all notebooks)
streamlit run app/app.py
```

## Methodology

### Feature Engineering — The 5 C's Framework

Rather than feeding all 150 raw columns into a model, features were selected and engineered based on how a credit analyst evaluates a borrower:

| Credit Factor | Features | Rationale |
|---|---|---|
| **Capacity** | DTI, pro forma DTI, payment-to-income ratio, loan-to-income, employment length | Can the borrower afford the payments after taking on this loan? |
| **Character** | FICO score, delinquencies, inquiries, public records, credit history length | Does the borrower's history suggest willingness to repay? |
| **Capital** | Revolving utilization, revolving balance-to-income, total leverage | Does the borrower have reserves and credit headroom? |
| **Conditions** | Interest rate, term, purpose, rate premium | What are the loan-specific risk factors? |

A key engineered feature is **pro forma DTI**, which incorporates the proposed loan's monthly payment into the borrower's existing debt obligations — the same calculation an underwriter would perform before approving a credit facility.

### Why Remove the Grade?

The initial XGBoost model achieved an AUC of 0.7275, but Lending Club's grade accounted for 68% of feature importance. This meant the model was primarily learning that Lending Club's existing risk assessment was a good predictor — a circular and uninteresting finding.

Removing grade and sub-grade forced the model to rely on borrower fundamentals. The AUC dropped by only 0.0018, revealing that the engineered features capture nearly the same risk signal independently. The production model uses no grade inputs and instead *generates* an implied grade from its prediction — a more useful and more honest tool.

### Evaluation — Beyond AUC

The model is evaluated not just on statistical metrics but on **business impact**:
- At each probability threshold, what percentage of defaults would be caught?
- How many good borrowers would be incorrectly declined?
- What is the estimated dollar impact of losses avoided?

This mirrors how credit policy decisions are actually made — it's a trade-off between risk appetite and origination volume.

## Limitations

- **Historical data (2007–2018)** — Credit conditions, underwriting standards, and macroeconomic environments have evolved since this dataset was compiled. Model performance on current vintages would need to be validated.

- **Survivorship bias** — The dataset only contains loans that were approved by Lending Club. Borrowers who were declined are absent, which means the model understates the true risk at the tails of the distribution. This is visible in the data: borrowers approved at extreme DTI levels often show lower-than-expected default rates because they had strong compensating factors that cleared Lending Club's screening.

- **Non-monotonic predictions** — XGBoost does not enforce monotonic relationships between features and predictions. This means small increases in loan amount can occasionally produce slightly lower default probabilities — a byproduct of how decision trees split data into discrete buckets. In a production credit environment, this is one reason logistic regression remains the industry standard for regulatory scorecards despite lower predictive power: it guarantees that more debt always equals more risk in the model output. XGBoost supports monotone constraints that could address this, but they were intentionally left off to preserve the model's natural behavior and highlight this trade-off.

- **Interest rate as a proxy** — In the no-grade model, interest rate is the top feature (25% importance). This is expected: Lending Club assigns higher rates to riskier borrowers, so the rate captures the platform's risk assessment as a continuous variable rather than a letter grade. The rate also has a real causal component — higher payments increase debt service burden — but the majority of its predictive power likely reflects the underlying risk pricing.

- **Self-reported income** — Income is not verified for all Lending Club loans, introducing noise into capacity-based features like DTI and payment-to-income.

- **Consumer lending only** — Commercial credit has different dynamics including cash flow analysis, collateral valuation, and guarantor structures that are not represented here.

## About

Built by **John McMillin**

MBA, Finance — University of Iowa Tippie College of Business  
BA, Finance and Real Estate — University of Northern Iowa  
CFA Level I | CBCA® | Google Data Analytics & Advanced Data Analytics Certificates

[LinkedIn](https://www.linkedin.com/in/johnmcmillin) | [IntrinsiQ Valuation Tool](https://intrinsiq-5rgzuqbxerfy89is5ywy36.streamlit.app/)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
