import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("ML DATA ANALYSIS - COMPREHENSIVE PROJECT")
print("=" * 60)

# ============================================================================
# 1. DATA LOADING & EXPLORATION
# ============================================================================
print("\n[1] LOADING DATA...")
print("-" * 40)

np.random.seed(42)
n_samples = 2000

data = pd.DataFrame({
    'id': range(1, n_samples + 1),
    'date': pd.date_range('2024-01-01', periods=n_samples, freq='h'),
    'price': np.random.randint(200000, 2000000, n_samples),
    'bedrooms': np.random.randint(1, 7, n_samples),
    'bathrooms': np.random.randint(1, 5, n_samples),
    'sqft_living': np.random.randint(800, 4500, n_samples),
    'sqft_lot': np.random.randint(3000, 15000, n_samples),
    'floors': np.random.randint(1, 4, n_samples),
    'waterfront': np.random.randint(0, 2, n_samples),
    'view': np.random.randint(0, 5, n_samples),
    'condition': np.random.randint(1, 6, n_samples),
    'grade': np.random.randint(3, 13, n_samples),
    'sqft_above': np.random.randint(500, 4000, n_samples),
    'sqft_basement': np.random.randint(0, 2000, n_samples),
    'yr_built': np.random.randint(1950, 2024, n_samples),
    'yr_renovated': np.random.choice([0, 1990, 2000, 2010, 2020], n_samples),
    'zipcode': np.random.randint(98001, 98100, n_samples),
    'lat': np.random.uniform(47.2, 47.8, n_samples),
    'long': np.random.uniform(-122.5, -121.5, n_samples),
    'sqft_living15': np.random.randint(1000, 4000, n_samples),
    'sqft_lot15': np.random.randint(3000, 15000, n_samples)
})

print(f"Dataset shape: {data.shape}")
print(f"\nFeatures:\n{data.columns.tolist()}")
print(f"Generated dataset: {data.shape[0]} samples, {data.shape[1]} features")

# ============================================================================
# 2. DATA PREPROCESSING & FEATURE ENGINEERING
# ============================================================================
print("\n[2] DATA PREPROCESSING & FEATURE ENGINEERING...")
print("-" * 40)

df = data.copy()

df['price_per_sqft'] = df['price'] / df['sqft_living']
df['total_rooms'] = df['bedrooms'] + df['bathrooms']
df['house_age'] = 2025 - df['yr_built']
df['renovated'] = (df['yr_renovated'] > 0).astype(int)
df['basement'] = (df['sqft_basement'] > 0).astype(int)

X = df.drop(['id', 'date', 'price'], axis=1)
y = df['price']

print(f"Engineered features: price_per_sqft, total_rooms, house_age, renovated, basement")
print(f"Feature matrix shape: {X.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# ============================================================================
# 3. SUPERVISED LEARNING - REGRESSION MODELS
# ============================================================================
print("\n[3] SUPERVISED LEARNING - REGRESSION...")
print("-" * 40)

results = {}

# Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
results['Linear Regression'] = {
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_lr)),
    'R2': r2_score(y_test, y_pred_lr),
    'MAE': mean_absolute_error(y_test, y_pred_lr)
}
print(f"Linear Regression - R2: {results['Linear Regression']['R2']:.4f}, RMSE: {results['Linear Regression']['RMSE']:.2f}")

# Ridge Regression (L2 regularization)
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
y_pred_ridge = ridge.predict(X_test_scaled)
results['Ridge'] = {
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_ridge)),
    'R2': r2_score(y_test, y_pred_ridge),
    'MAE': mean_absolute_error(y_test, y_pred_ridge)
}
print(f"Ridge - R2: {results['Ridge']['R2']:.4f}, RMSE: {results['Ridge']['RMSE']:.2f}")

# Lasso Regression (L1 regularization)
lasso = Lasso(alpha=1.0)
lasso.fit(X_train_scaled, y_train)
y_pred_lasso = lasso.predict(X_test_scaled)
results['Lasso'] = {
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_lasso)),
    'R2': r2_score(y_test, y_pred_lasso),
    'MAE': mean_absolute_error(y_test, y_pred_lasso)
}
print(f"Lasso - R2: {results['Lasso']['R2']:.4f}, RMSE: {results['Lasso']['RMSE']:.2f}")

# Decision Tree
dt = DecisionTreeRegressor(max_depth=10, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
results['Decision Tree'] = {
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_dt)),
    'R2': r2_score(y_test, y_pred_dt),
    'MAE': mean_absolute_error(y_test, y_pred_dt)
}
print(f"Decision Tree - R2: {results['Decision Tree']['R2']:.4f}, RMSE: {results['Decision Tree']['RMSE']:.2f}")

# K-Nearest Neighbors
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)
results['KNN'] = {
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_knn)),
    'R2': r2_score(y_test, y_pred_knn),
    'MAE': mean_absolute_error(y_test, y_pred_knn)
}
print(f"KNN - R2: {results['KNN']['R2']:.4f}, RMSE: {results['KNN']['RMSE']:.2f}")

# Random Forest
rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
results['Random Forest'] = {
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
    'R2': r2_score(y_test, y_pred_rf),
    'MAE': mean_absolute_error(y_test, y_pred_rf)
}
print(f"Random Forest - R2: {results['Random Forest']['R2']:.4f}, RMSE: {results['Random Forest']['RMSE']:.2f}")

# ============================================================================
# 4. NEURAL NETWORK (TensorFlow/Keras)
# ============================================================================
print("\n[4] NEURAL NETWORK (TensorFlow)...")
print("-" * 40)

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping

    nn_model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])

    nn_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    nn_model.fit(X_train_scaled, y_train, epochs=100, batch_size=32,
                validation_split=0.2, callbacks=[early_stop], verbose=0)

    y_pred_nn = nn_model.predict(X_test_scaled, verbose=0).flatten()
    results['Neural Network'] = {
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_nn)),
        'R2': r2_score(y_test, y_pred_nn),
        'MAE': mean_absolute_error(y_test, y_pred_nn)
    }
    print(f"Neural Network - R2: {results['Neural Network']['R2']:.4f}, RMSE: {results['Neural Network']['RMSE']:.2f}")
except Exception as e:
    print(f"TensorFlow error: {e}")

# ============================================================================
# 5. UNSUPERVISED LEARNING - CLUSTERING
# ============================================================================
print("\n[5] UNSUPERVISED LEARNING - CLUSTERING...")
print("-" * 40)

cluster_features = df[['sqft_living', 'bedrooms', 'bathrooms', 'grade', 'house_age']]
cluster_scaled = StandardScaler().fit_transform(cluster_features)

inertias = []
K_range = range(2, 10)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(cluster_scaled)
    inertias.append(kmeans.inertia_)

optimal_k = 4
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans_final.fit_predict(cluster_scaled)

df['cluster'] = clusters

print(f"Optimal clusters: {optimal_k}")
print(f"Cluster distribution:\n{pd.Series(clusters).value_counts().sort_index()}")
print(f"Cluster centers (sample):\n{kmeans_final.cluster_centers_[:2]}")

# ============================================================================
# 6. HYPERPARAMETER TUNING
# ============================================================================
print("\n[6] HYPERPARAMETER TUNING...")
print("-" * 40)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 15, 20],
    'min_samples_split': [2, 5, 10]
}

rf_tuned = RandomForestRegressor(random_state=42, n_jobs=-1)
grid_search = GridSearchCV(rf_tuned, param_grid, cv=3, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV R2: {grid_search.best_score_:.4f}")

y_pred_tuned = grid_search.predict(X_test)
results['Random Forest (Tuned)'] = {
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_tuned)),
    'R2': r2_score(y_test, y_pred_tuned),
    'MAE': mean_absolute_error(y_test, y_pred_tuned)
}
print(f"Random Forest (Tuned) - R2: {results['Random Forest (Tuned)']['R2']:.4f}")

# KNN hyperparameter tuning
print("\nKNN Tuning...")
k_values = range(3, 15)
cv_scores = []
for k in k_values:
    knn_temp = KNeighborsRegressor(n_neighbors=k)
    scores = cross_val_score(knn_temp, X_train_scaled, y_train, cv=5, scoring='r2')
    cv_scores.append(scores.mean())

best_k = k_values[np.argmax(cv_scores)]
print(f"Best K: {best_k} with CV R2: {max(cv_scores):.4f}")

# ============================================================================
# 7. MODEL EVALUATION SUMMARY
# ============================================================================
print("\n[7] MODEL EVALUATION SUMMARY...")
print("-" * 40)

results_df = pd.DataFrame(results).T
results_df = results_df.sort_values('R2', ascending=False)
print(f"\n{results_df}")

# Cross-validation for best model
cv_scores_rf = cross_val_score(rf, X_train, y_train, cv=5, scoring='r2')
print(f"\nRandom Forest CV R2: {cv_scores_rf.mean():.4f} (+/- {cv_scores_rf.std() * 2:.4f})")

# ============================================================================
# 8. FEATURE IMPORTANCE
# ============================================================================
print("\n[8] FEATURE IMPORTANCE...")
print("-" * 40)

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n{feature_importance.head(10)}")

# ============================================================================
# 9. CLASSIFICATION TASK (Derived from Regression)
# ============================================================================
print("\n[9] CLASSIFICATION TASK...")
print("-" * 40)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df_class = df.copy()
df_class['price_category'] = pd.qcut(df_class['price'], q=3, labels=['Low', 'Medium', 'High'])

X_class = df_class.drop(['id', 'date', 'price', 'price_category', 'price_per_sqft'], axis=1)
y_class = df_class['price_category']

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

scaler_c = StandardScaler()
X_train_c_scaled = scaler_c.fit_transform(X_train_c)
X_test_c_scaled = scaler_c.transform(X_test_c)

rf_class = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
rf_class.fit(X_train_c_scaled, y_train_c)
y_pred_class = rf_class.predict(X_test_c_scaled)

print(f"Classification Accuracy: {accuracy_score(y_test_c, y_pred_class):.4f}")
print(f"\nClassification Report:\n{classification_report(y_test_c, y_pred_class)}")

# ============================================================================
# RESULTS VISUALIZATION
# ============================================================================
print("\n[10] SAVING VISUALIZATIONS...")
print("-" * 40)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Model Comparison
ax1 = axes[0, 0]
models = list(results.keys())
r2_scores = [results[m]['R2'] for m in models]
ax1.barh(models, r2_scores, color='steelblue')
ax1.set_xlabel('R2 Score')
ax1.set_title('Model Comparison - R2 Score')

# Feature Importance
ax2 = axes[0, 1]
top_features = feature_importance.head(10)
ax2.barh(top_features['feature'], top_features['importance'], color='seagreen')
ax2.set_xlabel('Importance')
ax2.set_title('Feature Importance')

# Clustering
ax3 = axes[1, 0]
ax3.scatter(cluster_features['sqft_living'], cluster_features['bedrooms'], c=clusters, cmap='viridis', alpha=0.5)
ax3.scatter(kmeans_final.cluster_centers_[:, 0], kmeans_final.cluster_centers_[:, 1], c='red', marker='X', s=200)
ax3.set_xlabel('sqft_living')
ax3.set_ylabel('bedrooms')
ax3.set_title('K-Means Clustering')

# Elbow Method
ax4 = axes[1, 1]
ax4.plot(K_range, inertias, 'bo-')
ax4.set_xlabel('Number of Clusters (K)')
ax4.set_ylabel('Inertia')
ax4.set_title('Elbow Method for Optimal K')

plt.tight_layout()
plt.savefig('ml_analysis_results.png', dpi=150)
print("Saved: ml_analysis_results.png")

print("\n" + "=" * 60)
print("COMPLETED!")
print("=" * 60)