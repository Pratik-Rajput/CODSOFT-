# =============================================================
# TASK 1: TITANIC SURVIVAL PREDICTION
# CodSoft Data Science Internship
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)

import warnings
warnings.filterwarnings('ignore')

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print("=" * 60)
print("TITANIC SURVIVAL PREDICTION")
print("=" * 60)
print(f"\nDataset shape: {df.shape}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nColumn Info:")
print(df.info())
print(f"\nMissing Values:\n{df.isnull().sum()}")
print(f"\nSurvival Distribution:\n{df['Survived'].value_counts()}")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Titanic EDA', fontsize=16, fontweight='bold')

df['Survived'].value_counts().plot(kind='bar', ax=axes[0, 0],
    color=['salmon', 'steelblue'], edgecolor='black')
axes[0, 0].set_title('Survival Count')
axes[0, 0].set_xticklabels(['Not Survived', 'Survived'], rotation=0)
axes[0, 0].set_ylabel('Count')

pd.crosstab(df['Sex'], df['Survived']).plot(kind='bar', ax=axes[0, 1],
    color=['salmon', 'steelblue'], edgecolor='black')
axes[0, 1].set_title('Survival by Gender')
axes[0, 1].set_xticklabels(['Female', 'Male'], rotation=0)
axes[0, 1].legend(['Not Survived', 'Survived'])

pd.crosstab(df['Pclass'], df['Survived']).plot(kind='bar', ax=axes[0, 2],
    color=['salmon', 'steelblue'], edgecolor='black')
axes[0, 2].set_title('Survival by Passenger Class')
axes[0, 2].set_xticklabels(['1st', '2nd', '3rd'], rotation=0)
axes[0, 2].legend(['Not Survived', 'Survived'])

df['Age'].dropna().plot(kind='hist', bins=30, ax=axes[1, 0],
    color='steelblue', edgecolor='black')
axes[1, 0].set_title('Age Distribution')
axes[1, 0].set_xlabel('Age')

df['Fare'].plot(kind='hist', bins=40, ax=axes[1, 1],
    color='salmon', edgecolor='black')
axes[1, 1].set_title('Fare Distribution')
axes[1, 1].set_xlabel('Fare')

pd.crosstab(df['Embarked'], df['Survived']).plot(kind='bar', ax=axes[1, 2],
    color=['salmon', 'steelblue'], edgecolor='black')
axes[1, 2].set_title('Survival by Embarkation Port')
axes[1, 2].legend(['Not Survived', 'Survived'])

plt.tight_layout()
plt.savefig('titanic_eda.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n[EDA plots saved as titanic_eda.png]")

print("\n" + "=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

df_model = df.copy()

df_model['Title'] = df_model['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
title_map = {
    'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
    'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
    'Mlle': 'Miss', 'Countess': 'Rare', 'Ms': 'Miss', 'Lady': 'Rare',
    'Jonkheer': 'Rare', 'Don': 'Rare', 'Dona': 'Rare', 'Mme': 'Mrs',
    'Capt': 'Rare', 'Sir': 'Rare'
}
df_model['Title'] = df_model['Title'].map(title_map).fillna('Rare')
print(f"Titles extracted:\n{df_model['Title'].value_counts()}")

df_model['Age'] = df_model.groupby('Title')['Age'].transform(
    lambda x: x.fillna(x.median())
)

df_model['Embarked'] = df_model['Embarked'].fillna(df_model['Embarked'].mode()[0])

df_model['Fare'] = df_model['Fare'].fillna(df_model['Fare'].median())

df_model['FamilySize'] = df_model['SibSp'] + df_model['Parch'] + 1
df_model['IsAlone'] = (df_model['FamilySize'] == 1).astype(int)

df_model['AgeBin'] = pd.cut(df_model['Age'],
    bins=[0, 12, 18, 35, 60, 100],
    labels=['Child', 'Teen', 'Adult', 'MiddleAge', 'Senior'])

df_model['FareBin'] = pd.qcut(df_model['Fare'], q=4,
    labels=['Low', 'Mid', 'High', 'VeryHigh'])

drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Age', 'Fare',
             'SibSp', 'Parch']
df_model.drop(columns=drop_cols, inplace=True)

le = LabelEncoder()
cat_cols = ['Sex', 'Embarked', 'Title', 'AgeBin', 'FareBin']
for col in cat_cols:
    df_model[col] = le.fit_transform(df_model[col].astype(str))

print(f"\nFinal features:\n{df_model.columns.tolist()}")
print(f"\nProcessed dataset shape: {df_model.shape}")
print(f"\nMissing values after preprocessing: {df_model.isnull().sum().sum()}")


X = df_model.drop('Survived', axis=1)
y = df_model['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

print("\n" + "=" * 60)
print("MODEL TRAINING & EVALUATION")
print("=" * 60)

models = {
    "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    X_tr = X_train_scaled if name == "Logistic Regression" else X_train
    X_te = X_test_scaled  if name == "Logistic Regression" else X_test

    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    roc  = roc_auc_score(y_test, y_prob)
    cv   = cross_val_score(model, X_tr, y_train, cv=5, scoring='accuracy').mean()

    results[name] = {'Accuracy': acc, 'ROC-AUC': roc, 'CV Accuracy': cv,
                     'model': model, 'y_pred': y_pred, 'y_prob': y_prob,
                     'X_te': X_te}

    print(f"\n--- {name} ---")
    print(f"  Accuracy       : {acc:.4f}")
    print(f"  ROC-AUC        : {roc:.4f}")
    print(f"  CV Accuracy    : {cv:.4f}")
    print(f"\n  Classification Report:\n{classification_report(y_test, y_pred, target_names=['Not Survived', 'Survived'])}")


best_name = max(results, key=lambda k: results[k]['ROC-AUC'])
best      = results[best_name]
print(f"\n{'=' * 60}")
print(f"BEST MODEL: {best_name}  (ROC-AUC = {best['ROC-AUC']:.4f})")
print(f"{'=' * 60}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f'Best Model: {best_name}', fontsize=14, fontweight='bold')

cm = confusion_matrix(y_test, best['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Not Survived', 'Survived'],
            yticklabels=['Not Survived', 'Survived'])
axes[0].set_title('Confusion Matrix')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')

for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    axes[1].plot(fpr, tpr, label=f"{name} (AUC={res['ROC-AUC']:.3f})")
axes[1].plot([0, 1], [0, 1], 'k--', label='Random')
axes[1].set_title('ROC Curves')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].legend(loc='lower right')

plt.tight_layout()
plt.savefig('titanic_model_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("[Model result plots saved as titanic_model_results.png]")

rf_model  = results['Random Forest']['model']
feat_imp  = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 5))
feat_imp.plot(kind='bar', color='steelblue', edgecolor='black')
plt.title('Feature Importances (Random Forest)')
plt.ylabel('Importance Score')
plt.tight_layout()
plt.savefig('titanic_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print("[Feature importance plot saved as titanic_feature_importance.png]")


print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
summary_df = pd.DataFrame({
    name: {
        'Accuracy': f"{res['Accuracy']:.4f}",
        'ROC-AUC':  f"{res['ROC-AUC']:.4f}",
        'CV Acc':   f"{res['CV Accuracy']:.4f}"
    }
    for name, res in results.items()
}).T
print(summary_df.to_string())
print(f"\n Best Model  : {best_name}")
print(f" Best ROC-AUC: {best['ROC-AUC']:.4f}")
print(f" Best Acc    : {best['Accuracy']:.4f}")