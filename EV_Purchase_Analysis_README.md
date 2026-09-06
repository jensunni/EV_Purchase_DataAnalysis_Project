# EV Purchase Analysis — Understanding the Factors Influencing Electric Vehicle Adoption

*All numbers in this document are computed directly from `EV_Adoption_and_Range_Anxiety_Dataset.csv` — verified, not estimated.*

## 1. Business Problem

An EV company wants to understand which customer characteristics and concerns influence a person's decision to purchase an electric vehicle. By analyzing customer demographics, income, commuting distance, environmental concerns, and range anxiety, the company can identify potential EV buyers and improve its marketing strategy.

## 2. Objective

- Identify which customer segments are most likely to buy an EV
- Quantify the effect of income, subsidies, and range anxiety on purchase intent
- Test whether charging access and demographics actually move the needle, or just seem like they should
- Produce a short set of actionable customer-targeting recommendations

## 3. Dataset

**File:** `EV_Adoption_and_Range_Anxiety_Dataset.csv`
**Rows:** 10,000 customers, 15 columns, 0 duplicates, `Buyer_ID` confirmed unique

| Column | Type | Values / Range |
|---|---|---|
| `Buyer_ID` | Identifier | EV00001–EV10000 |
| `Age` | Numeric | 25–69 |
| `Gender` | Categorical | Male / Female / Other |
| `Annual_Income_USD` | Numeric | 30,000–223,345 (178 missing → filled with median) |
| `City_Type` | Categorical | Urban / Suburban / Rural |
| `Daily_Commute_km` | Numeric | 5–135.5 (181 missing → filled with median) |
| `Number_of_Cars_Owned` | Numeric | 1–4 |
| `Current_Car_Type` | Categorical | Sedan / SUV / Truck / Hatchback |
| `Charging_Stations_Near_Home` | Numeric | count |
| `Charging_Stations_Near_Work` | Numeric | count |
| `Home_Charging_Possible` | Categorical | **Yes / No** (text, not 0/1) |
| `Environmental_Concern_Level` | Numeric | 1–5 (184 missing → filled with median) |
| `Subsidy_Available` | Categorical | **Yes / No** (text, not 0/1) |
| `Range_Anxiety_Level` | Categorical | **Low / Medium / High** (ordinal text) |
| `Will_Buy_EV` | **Target** | **Yes / No** (text, not 0/1) |

**Text to numeric for some column for calculation:** `Will_Buy_EV`, `Home_Charging_Possible`, and `Subsidy_Available` are stored as the text `"Yes"`/`"No"`, not `1`/`0`. `Range_Anxiety_Level` is text (`Low`/`Medium`/`High`), not a number. Any correlation, `groupby().mean()`, or model training step must map these to numeric first:
```python
df['Buy_Flag'] = (df['Will_Buy_EV'] == 'Yes').astype(int)
df['Subsidy_Flag'] = (df['Subsidy_Available'] == 'Yes').astype(int)
df['HomeCharge_Flag'] = (df['Home_Charging_Possible'] == 'Yes').astype(int)
df['Range_Anxiety_Num'] = df['Range_Anxiety_Level'].map({'Low': 0, 'Medium': 1, 'High': 2})
```

**Target balance:** 1,750 of 10,000 (17.5%) are "Yes" — imbalanced, so accuracy alone is misleading later.

## 4. Data Cleaning

| Column | Missing | Fix |
|---|---|---|
| `Annual_Income_USD` | 178 | Filled with median |
| `Daily_Commute_km` | 181 | Filled with median |
| `Environmental_Concern_Level` | 184 | Filled with median |

## 5. Exploratory Data Analysis — Verified Findings

| # | Question | Verified Result |
|---|---|---|
| 1 | How many customers will buy Electrical Vehicle(EV)? | **1,750 / 10,000 (17.5%)** |
| 2 | Which age group buys most? | Flat across all groups (16.6%–19.0%) — **age is a weak predictor** |
| 3 | Does income differ? | Median income: **$95,716 (buyers) vs. $82,612 (non-buyers)** — a real, meaningful gap |
| 4 | Does home charging matter? | **20.1% buy rate (Yes) vs. 13.4% (No)** — a real but moderate effect |
| 5 | Does range anxiety matter? | **Low: 19.9%, Medium: 6.4%, High: 0.6%** — one of the strongest effects in the dataset |
| 6 | Does gender matter? | Female 18.0%, Male 17.1%, Other 16.6% — **negligible difference** |
| 7 | Does city type matter? | Rural 19.0%, Suburban 17.7%, Urban 16.9% — **small effect** |
| 8 | Does commute distance matter? | Median 39.4 km (buyers) vs. 40.2 km (non-buyers) — **essentially no effect** |
| 9 | Does number of cars owned matter? | 14.9%–18.7% across 1–4 cars — **no clear pattern** |
| 10 | Does current car type matter? | SUV 18.1%, Hatchback 17.5%, Sedan 17.2%, Truck 16.5% — **negligible difference** |
| 11 | Do nearby home charging stations matter? | 5.32 (buyers) vs. 5.35 (non-buyers) avg count — **no effect** |
| 12 | Do nearby work charging stations matter? | 7.47 (buyers) vs. 7.46 (non-buyers) avg count — **no effect** |
| 13 | Does subsidy availability matter? | **27.4% buy rate (subsidy) vs. 2.5% (no subsidy) — an 11x difference, by far the single strongest factor** |
| 14 | Does environmental concern matter? | Mean score 4.09 (buyers) vs. 2.75 (non-buyers) on a 1–5 scale — **very strong effect** |

**Correction to earlier assumption:** charging-station proximity (near home or work) was assumed to be a likely strong predictor before this data was available. The real numbers show **no measurable effect at all** — don't lead a presentation with this factor.

## 6. Correlation Ranking (with target properly encoded)

```python
numeric_cols = ['Age','Annual_Income_USD','Daily_Commute_km','Number_of_Cars_Owned',
                 'Charging_Stations_Near_Home','Charging_Stations_Near_Work',
                 'Environmental_Concern_Level','Range_Anxiety_Num','Subsidy_Flag','HomeCharge_Flag']
df[numeric_cols + ['Buy_Flag']].corr()['Buy_Flag'].sort_values(key=abs, ascending=False)
```

| Feature | Correlation with `Buy_Flag` |
|---|---|
| Environmental_Concern_Level | **+0.363** |
| Subsidy_Flag | **+0.321** |
| Annual_Income_USD | +0.173 |
| Range_Anxiety_Num | −0.138 |
| HomeCharge_Flag | +0.086 |
| Daily_Commute_km | −0.027 |
| Age | +0.004 |
| Charging_Stations_Near_Home | −0.003 |
| Charging_Stations_Near_Work | +0.001 |
| Number_of_Cars_Owned | +0.0004 |

Four factors matter (Environmental Concern, Subsidy, Income, Range Anxiety); everything else in this dataset is essentially noise.

## 7. Machine Learning

**Target:** `Will_Buy_EV` mapped to 0/1. **Model:** Logistic Regression with `class_weight='balanced'` (needed — see below). **Features:** all columns except `Buyer_ID`. Categorical columns one-hot encoded.

| Metric | Value |
|---|---|
| Baseline (majority class, always predict "No") | 82.5% |
| Model accuracy | 81.4% |
| Precision | 0.48 |
| Recall | **0.86** |
| F1-score | 0.62 |
| ROC-AUC | **0.91** |

**Why accuracy looks *worse* than the baseline here:** this is expected and worth explaining, not hiding. With `class_weight='balanced'`, the model deliberately trades some accuracy for much better recall (86% of actual buyers correctly identified, vs. 0% for a model that just predicts "No" every time). ROC-AUC of 0.91 shows the model separates buyers from non-buyers very well overall — accuracy alone would have hidden that.

**Top coefficients (what actually drives the prediction):**
- Not having a subsidy is the single strongest negative driver
- Medium/High range anxiety strongly reduces predicted purchase likelihood
- No home charging access reduces it further
- Higher environmental concern score increases it
- Gender, city type, and current car type all had small, largely offsetting effects — consistent with the near-zero correlations in Section 6

## 8. Business Recommendations

**Finding:** Subsidy availability is an 11x swing factor (2.5% → 27.4% buy rate).
**Recommendation:** Lead marketing and sales conversations with subsidy eligibility — it's the single highest-leverage lever in this dataset, well ahead of demographic targeting.

**Finding:** High range anxiety almost completely suppresses purchase intent (0.6% buy rate vs. 19.9% for low anxiety).
**Recommendation:** Invest in range-anxiety-reducing messaging (warranty, roadside assistance, range-calculator tools) before spending on broad demographic campaigns.

**Finding:** Charging-station proximity (home or work) showed no measurable relationship with purchase intent in this dataset.
**Recommendation:** Don't prioritize "chargers per neighborhood" as a marketing signal — the data doesn't support it here, even though it's an intuitive assumption.

**Finding:** Customers with high environmental concern, subsidy access, and home charging together buy at 46.5% — versus 17.5% overall.
**Recommendation:** Use this three-factor combination as a lead-scoring rule for the sales team, rather than any single factor alone.

## 9. Future Improvements

- Compare Logistic Regression against Random Forest / Gradient Boosting to see if the non-linear model captures interaction effects (e.g., subsidy × income) better
- Try SMOTE or threshold-tuning as an alternative to `class_weight='balanced'` and compare precision/recall trade-offs
- Investigate why Medium range anxiety (6.4%) sits so much closer to High (0.6%) than to Low (19.9%) — possible non-linear cutoff effect worth a dedicated chart

## 10. Tools Used

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

## 11. Project Structure

```
ev-purchase-analysis/
  |-- data/EV_Adoption_and_Range_Anxiety_Dataset.csv
  |-- notebooks/EV_Purchase_Analysis.ipynb
  |-- EV_Purchase_Analysis_README.md
```