
# TASK 3: IRIS FLOWER CLASSIFICATION

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)

import warnings
warnings.filterwarnings('ignore')

iris    = load_iris()
X       = pd.DataFrame(iris.data,   columns=iris.feature_names)
y       = pd.Series(iris.target)
species = iris.target_names         

print("=" * 55)
print("IRIS FLOWER CLASSIFICATION")
print("=" * 55)
print(f"\nDataset shape : {X.shape}")
print(f"Classes       : {list(species)}")
print(f"\nFirst 5 rows:\n{X.head()}")
print(f"\nClass distribution:\n{y.value_counts().rename({0:'setosa',1:'versicolor',2:'virginica'})}")
print(f"\nMissing values: {X.isnull().sum().sum()}")   # always 0 for this dataset

df_eda        = X.copy()
df_eda['species'] = y.map({0:'setosa', 1:'versicolor', 2:'virginica'})

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle('Iris EDA — Feature Distributions by Species', fontsize=14, fontweight='bold')

colors = ['steelblue', 'salmon', 'seagreen']
for ax, feature in zip(axes.flatten(), iris.feature_names):
    for i, sp in enumerate(species):
        subset = df_eda[df_eda['species'] == sp][feature]
        ax.hist(subset, bins=15, alpha=0.6, label=sp, color=colors[i], edgecolor='black')
    ax.set_title(feature)
    ax.set_xlabel('cm')
    ax.legend()

plt.tight_layout()
plt.savefig('iris_eda.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n[EDA saved as iris_eda.png]")

sns.pairplot(df_eda, hue='species', palette={'setosa':'steelblue',
             'versicolor':'salmon', 'virginica':'seagreen'}, diag_kind='hist')
plt.suptitle('Iris Pairplot', y=1.02, fontsize=13, fontweight='bold')
plt.savefig('iris_pairplot.png', dpi=150, bbox_inches='tight')
plt.show()
print("[Pairplot saved as iris_pairplot.png]")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler       = StandardScaler()
X_train_sc   = scaler.fit_transform(X_train)
X_test_sc    = scaler.transform(X_test)

print(f"\nTrain: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

models = {
    "Logistic Regression": LogisticRegression(max_iter=200, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}
print("\n" + "=" * 55)
for name, model in models.items():
    Xtr = X_train_sc if name == "Logistic Regression" else X_train
    Xte = X_test_sc  if name == "Logistic Regression" else X_test

    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    acc    = accuracy_score(y_test, y_pred)
    cv     = cross_val_score(model, Xtr, y_train, cv=5).mean()

    results[name] = {'Accuracy': acc, 'CV Accuracy': cv,
                     'model': model, 'y_pred': y_pred, 'Xte': Xte}

    print(f"\n--- {name} ---")
    print(f"  Test Accuracy : {acc:.4f}")
    print(f"  CV Accuracy   : {cv:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=species)}")

best_name = max(results, key=lambda k: results[k]['Accuracy'])
print(f"\nBest Model: {best_name}  (Accuracy = {results[best_name]['Accuracy']:.4f})")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f'Results — {best_name}', fontsize=13, fontweight='bold')

cm = confusion_matrix(y_test, results[best_name]['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=species, yticklabels=species)
axes[0].set_title('Confusion Matrix')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')

rf        = results['Random Forest']['model']
feat_imp  = pd.Series(rf.feature_importances_, index=iris.feature_names).sort_values(ascending=False)
feat_imp.plot(kind='bar', ax=axes[1], color='steelblue', edgecolor='black')
axes[1].set_title('Feature Importances (Random Forest)')
axes[1].set_ylabel('Importance Score')
axes[1].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig('iris_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("[Results saved as iris_results.png]")

sample       = np.array([[5.1, 3.5, 1.4, 0.2]])   
sample_sc    = scaler.transform(sample)
rf_pred      = results['Random Forest']['model'].predict(sample)
lr_pred      = results['Logistic Regression']['model'].predict(sample_sc)

print("\n" + "=" * 55)
print("CUSTOM PREDICTION  [5.1 cm, 3.5 cm, 1.4 cm, 0.2 cm]")
print("=" * 55)
print(f"  Random Forest      → {species[rf_pred[0]]}")
print(f"  Logistic Regression→ {species[lr_pred[0]]}")

print("\n" + "=" * 55)
print("FINAL SUMMARY")
print("=" * 55)
summary = pd.DataFrame({
    name: {'Test Acc': f"{res['Accuracy']:.4f}",
           'CV Acc':   f"{res['CV Accuracy']:.4f}"}
    for name, res in results.items()
}).T
print(summary.to_string())
print(f"\n Best Model : {best_name}")
print(f" Accuracy   : {results[best_name]['Accuracy']:.4f}")