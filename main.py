import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report


# Effects of COVID on Daily Life (survey dataset)
data = pd.read_csv("psyco.csv").copy()
data["age"] = data["age"].replace("Dec-18", "0-18")

# Clean up messy columns (extra commas / unnamed cols / duplicates)
data = data.loc[:, ~data.columns.astype(str).str.contains("^Unnamed")]
data.columns = [str(c).strip() for c in data.columns]
data = data.loc[:, ~data.columns.duplicated()]


# -----------------------------
#         EDA / Visuals 
# -----------------------------

# Graph of Average working hours before and during the pandemic by occupation
time_spent_graph = data[["occupation", "time_bp", "time_dp"]].copy()
time_spent_graph = time_spent_graph.groupby("occupation", as_index=True).mean()
time_spent_graph.plot(kind="bar", figsize=(10, 6), color=["blue", "green"])
plt.xlabel("Occupation")
plt.ylabel("Average working hours")
plt.title("Average working hours before and during the pandemic by occupation")
plt.xticks(rotation=45)
plt.legend(["Time Spent on work before the pandemic", "Time spent on work during the pandemic"])
plt.tight_layout()
# plt.show()

# Graph of work type preferred by gender
plt.figure(figsize=(9, 5))
sns.countplot(x="prefer", hue="gender", data=data, palette=["blue", "pink", "red"])
plt.title("Preference for location of work by gender")
plt.xlabel("Preference to work from home or in person")
plt.ylabel("Person Count")
plt.legend(["Male", "Female", "Prefer not to say"])
# plt.show()

# Graph of work type preferred by occupation
plt.figure(figsize=(10, 5))
sns.countplot(x="prefer", hue="occupation", data=data, palette="viridis")
plt.title("Preference for location of work by occupation")
plt.xlabel("Preference to work from home or in person")
plt.ylabel("Person Count")
plt.legend(title="Occupation")
plt.tight_layout()
plt.show()

# Graph of preferred work type by age
age_vs_work = data[["age", "prefer"]].copy()
grouped_data = age_vs_work.groupby(["age", "prefer"]).size().unstack(fill_value=0)
grouped_data.plot(kind="bar", stacked=True)
plt.xlabel("Age")
plt.ylabel("Count")
plt.title("Preferred Work Type by Age")
plt.legend(title="Preferred Work Type")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Graph of if working from home is needed with age
age_vs_workingfromhome = data[["age", "certaindays_hw"]].copy()
grouped_data = age_vs_workingfromhome.groupby(["age", "certaindays_hw"]).size().unstack(fill_value=0)
grouped_data.plot(kind="bar", stacked=True)
plt.xlabel("Age")
plt.ylabel("Count")
plt.title("If Working From Home on Certain Days Is Needed by Age")
plt.legend(title="Work From Home Needed by Age")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Travel and work hours before and during pandemic by occupation
travel_work = data[["occupation", "time_bp", "time_dp", "travel_time"]].copy()
travel_work["total_time_bp"] = travel_work["time_bp"] + travel_work["travel_time"]
travel_work["total_time_dp"] = travel_work["time_dp"] + travel_work["travel_time"]
travel_work = travel_work.groupby("occupation", as_index=True).mean()

travel_work[["total_time_bp", "total_time_dp"]].plot(kind="bar", figsize=(10, 6), color=["blue", "green"])
plt.xlabel("Occupation")
plt.ylabel("Total Hours (Work + Travel)")
plt.title("Total Working Hours (Work + Travel) Before and During Pandemic by Occupation")
plt.xticks(rotation=45)
plt.legend(["Before Pandemic", "During Pandemic"])
plt.tight_layout()
plt.show()

# Home environment and productivity change
home_prod = data[["home_env", "prod_inc"]].copy()
home_prod_grouped = home_prod.groupby(["home_env", "prod_inc"]).size().unstack(fill_value=0)
home_prod_grouped.plot(kind="bar", stacked=True)
plt.xlabel("Home Environment")
plt.ylabel("Count")
plt.title("Impact of Home Environment on Productivity")
plt.legend(title="Productivity Increase")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# Logistic Regression: predict preference for WFH (binary)
# ---------------------------------------------------------

# Prefer WFH if "home" appears in the preference text (works for "Work/study from home")
data["prefer_wfh"] = data["prefer"].astype(str).str.contains("home", case=False, na=False).astype(int)

feature_cols = [
    "gender", "age", "occupation", "line_of_work",
    "time_bp", "time_dp", "travel_time",
    "easeof_online", "home_env", "prod_inc", "sleep_bal", "new_skill",
    "fam_connect", "relaxed", "self_time",
    "certaindays_hw"
]
feature_cols = [c for c in feature_cols if c in data.columns]

df = data.dropna(subset=feature_cols + ["prefer"]).copy()

X = df[feature_cols]
y = df["prefer_wfh"]

numeric_features = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
categorical_features = [c for c in X.columns if c not in numeric_features]

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ],
    remainder="drop"
)

clf = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)

print("\n--- Logistic Regression Results (Prefer WFH vs Not) ---")
print(f"Dataset used for model: n={len(df)} rows | {X.shape[1]} raw features")
print(f"Accuracy: {acc:.3f} | F1: {f1:.3f} | ROC-AUC: {auc:.3f}")
print("Confusion matrix:\n", cm)
print("\nClassification report:\n", classification_report(y_test, y_pred))

# Top drivers (by absolute coefficient)
ohe = clf.named_steps["preprocess"].named_transformers_["cat"]
feature_names = []
feature_names += numeric_features
if categorical_features:
    feature_names += ohe.get_feature_names_out(categorical_features).tolist()

coefs = clf.named_steps["model"].coef_[0]
coef_df = pd.DataFrame({"feature": feature_names, "coef": coefs})
coef_df["abs_coef"] = coef_df["coef"].abs()

print("\nTop drivers (by |coef|):")
print(coef_df.sort_values("abs_coef", ascending=False).head(10)[["feature", "coef"]])
