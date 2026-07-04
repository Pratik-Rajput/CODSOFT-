
# TASK 2: MOVIE RATING PREDICTION WITH PYTHON

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import warnings
warnings.filterwarnings('ignore')

URL = "https://raw.githubusercontent.com/rashida048/Datasets/master/movie_dataset.csv"
df  = pd.read_csv(URL)

print("Shape:", df.shape)
print(df[['title', 'genres', 'director', 'runtime', 'budget', 'vote_count', 'vote_average']].head())
print("\nMissing values:\n", df[['genres','director','cast','runtime','budget','vote_average']].isnull().sum())

df = df[['title', 'genres', 'director', 'cast',
         'runtime', 'budget', 'vote_count', 'vote_average']].copy()

df.dropna(subset=['vote_average'], inplace=True)

df['genres'].fillna('Unknown', inplace=True)
df['director'].fillna('Unknown', inplace=True)
df['cast'].fillna('Unknown', inplace=True)
df['runtime'].fillna(df['runtime'].median(), inplace=True)
df['budget'].fillna(0, inplace=True)

print("\nAfter cleaning — shape:", df.shape)
print("Nulls remaining:", df.isnull().sum().sum())

df['primary_genre'] = df['genres'].str.split().str[0]

df['lead_actor'] = df['cast'].str.split().str[0].fillna('Unknown')

le = LabelEncoder()
df['genre_enc']    = le.fit_transform(df['primary_genre'])
df['director_enc'] = le.fit_transform(df['director'])
df['actor_enc']    = le.fit_transform(df['lead_actor'])

features = ['runtime', 'budget', 'vote_count',
            'genre_enc', 'director_enc', 'actor_enc']
X = df[features]
y = df['vote_average']

print(f"\nFeatures: {features}")
print(f"X: {X.shape}  |  y range: {y.min():.1f} – {y.max():.1f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Movie Rating EDA', fontsize=14, fontweight='bold')

y.plot(kind='hist', bins=30, ax=axes[0], color='steelblue', edgecolor='black')
axes[0].set_title('Rating Distribution')
axes[0].set_xlabel('Rating')

axes[1].scatter(df['vote_count'], y, alpha=0.3, color='salmon')
axes[1].set_title('Vote Count vs Rating')
axes[1].set_xlabel('Vote Count')
axes[1].set_ylabel('Rating')

top_genres = df.groupby('primary_genre')['vote_average'].mean().sort_values(ascending=False).head(10)
top_genres.plot(kind='bar', ax=axes[2], color='steelblue', edgecolor='black')
axes[2].set_title('Avg Rating by Genre (Top 10)')
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('movie_eda.png', dpi=150, bbox_inches='tight')
plt.show()
print("[EDA saved as movie_eda.png]")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

models = {
    "Linear Regression":  LinearRegression(),
    "Random Forest":      RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting":  GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    results[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2,
                     'model': model, 'y_pred': y_pred}

    print(f"\n--- {name} ---")
    print(f"  MAE  : {mae:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  R²   : {r2:.4f}")

best_name = max(results, key=lambda k: results[k]['R2'])
best = results[best_name]
print(f"\nBest Model: {best_name}  (R² = {best['R2']:.4f})")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f'Best Model: {best_name}', fontsize=13, fontweight='bold')

axes[0].scatter(y_test, best['y_pred'], alpha=0.4, color='steelblue')
axes[0].plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0].set_xlabel('Actual Rating')
axes[0].set_ylabel('Predicted Rating')
axes[0].set_title('Actual vs Predicted')

rf = results['Random Forest']['model']
feat_imp = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
feat_imp.plot(kind='bar', ax=axes[1], color='salmon', edgecolor='black')
axes[1].set_title('Feature Importances (Random Forest)')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('movie_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("[Results saved as movie_results.png]")

print("\n" + "=" * 50)
print("FINAL SUMMARY")
print("=" * 50)
summary = pd.DataFrame({
    name: {'MAE': f"{res['MAE']:.4f}", 'RMSE': f"{res['RMSE']:.4f}", 'R²': f"{res['R2']:.4f}"}
    for name, res in results.items()
}).T
print(summary.to_string())
print(f"\n Best Model : {best_name}")
print(f" R²         : {best['R2']:.4f}")
print(f" MAE        : {best['MAE']:.4f}")