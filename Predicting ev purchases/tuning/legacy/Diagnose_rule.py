import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from features_ev import load_raw

tr, _ = load_raw()
y = (tr["Will_Buy_EV"] == "Yes").astype(int).to_numpy()
real = tr["Annual_Income_USD"] != 30000          # rows with a genuine income

# --- step 1: let LR recover the buy-rule weights ---
drivers = pd.DataFrame({
    "income":   tr["Annual_Income_USD"] / 1e5,
    "env":      tr["Environmental_Concern_Level"],
    "subsidy":  (tr["Subsidy_Available"] == "Yes").astype(float),
    "anx_med":  (tr["Range_Anxiety_Level"] == "Medium").astype(float),
    "anx_high": (tr["Range_Anxiety_Level"] == "High").astype(float),
})
spl = StratifiedKFold(5, shuffle=True, random_state=42)
p = cross_val_predict(LogisticRegression(max_iter=1000), drivers, y,
                      cv=spl, method="predict_proba")[:, 1]
print("recovered-rule OOF AUC:", roc_auc_score(y, p))
print("fitted weights:", dict(zip(drivers.columns,
      LogisticRegression(max_iter=1000).fit(drivers, y).coef_[0])))

# --- step 2: functional-form check for one driver ---
b = pd.qcut(tr.loc[real, "Daily_Commute_km"], 50, duplicates='drop')
rate = y[real.to_numpy()].astype(float)
print(pd.Series(rate).groupby(b.to_numpy()).mean())