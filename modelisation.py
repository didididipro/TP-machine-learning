# --- 1. Imports ---
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import joblib


print("Imports finished.")

# --- 2. Upload your dataset ---
uploaded = files.upload()

# --- 3. Load and preprocess data ---
data = pd.read_csv("colon_cancer.csv", sep=";")
data["tissue_status"] = data["tissue_status"].map({"normal": 0, "tumoral": 1})

# Drop missing values if any
if data.isnull().sum().sum() > 0:
    data = data.dropna()

X = data.drop(columns=["id_sample", "tissue_status"])
y = data["tissue_status"]

# --- 4. Gene selection based on p-value ---
p_values = {}
for gene in X.columns:
    normal = X[gene][y == 0]
    tumoral = X[gene][y == 1]
    t_stat, p_val = ttest_ind(normal, tumoral, equal_var=False)
    p_values[gene] = p_val

# Find best gene
best_gene = min(p_values, key=p_values.get)
print(f"Best gene selected: {best_gene} (p-value: {p_values[best_gene]:.6f})")

# Prepare data using only the best gene
X_best = X[[best_gene]]

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_best)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# --- 5. Model testing ---
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42),
    "SVM": SVC(random_state=42),
    "KNN": KNeighborsClassifier(),
}

model_scores = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    mean_score = scores.mean()
    model_scores[name] = mean_score
    print(f"{name} - Mean accuracy: {mean_score:.4f}")

# Select best model
best_model_name = max(model_scores, key=model_scores.get)
best_score = model_scores[best_model_name]
print(f"Best model: {best_model_name} with accuracy {best_score:.4f}")

# --- 6. Train best model ---
best_model = models[best_model_name]
best_model.fit(X_train, y_train)

# Evaluate on test set
test_score = best_model.score(X_test, y_test)
print(f"Test accuracy: {test_score:.4f}")

# --- 7. Save model and scaler ---
joblib.dump(best_model, "colon_cancer_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Model and scaler exported: colon_cancer_model.pkl and scaler.pkl")
