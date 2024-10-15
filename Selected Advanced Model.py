# Adjusted Code with More Efficient Settings

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from joblib import parallel_backend, dump
import time
import psutil
import os
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# Load prepared data
X_train = pd.read_csv(r'D:\Project Microsoft\Train_split.csv')
y_train = X_train.pop('IncidentGrade')
X_val = pd.read_csv(r'D:\Project Microsoft\Valid_split.csv')
y_val = X_val.pop('IncidentGrade')

# Use a smaller subset for initial hyperparameter tuning (20% of data)
X_sample = X_train.sample(frac=0.2, random_state=42)
y_sample = y_train.loc[X_sample.index]

# Initialize SMOTE
smote = SMOTE(random_state=42)

# Initialize Random Forest classifier
rf = RandomForestClassifier(random_state=42)

# Simplified parameter grid for RandomizedSearchCV
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 20, 30],
    'max_features': ['sqrt', None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True]
}

# Setup randomized search with fewer cross-validation folds
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
random_search = RandomizedSearchCV(
    rf,
    param_distributions=param_dist,
    n_iter=20,  # Further reduced number of parameter settings
    cv=cv,
    scoring='f1_macro',
    n_jobs=-1,
    random_state=42
)

# Create pipeline with SMOTE and RandomizedSearchCV
pipeline = Pipeline([
    ('smote', smote),
    ('random_search', random_search)
])

# Train model with the sampled data and time tracking
start_time = time.time()
with parallel_backend('loky'):
    pipeline.fit(X_sample, y_sample)
end_time = time.time()

# Get memory usage
process = psutil.Process(os.getpid())
memory_usage = process.memory_info().rss / (1024 ** 2)  # Convert to MB

# Get the best model
best_rf = pipeline.named_steps['random_search'].best_estimator_

# Evaluate the model on the full validation data
y_pred = best_rf.predict(X_val)

# Print results
print(f"Best Hyperparameters: {pipeline.named_steps['random_search'].best_params_}")
print("Classification Report:")
print(classification_report(y_val, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_val, y_pred))
print(f"Training Time: {end_time - start_time:.2f} seconds")
print(f"Memory Usage: {memory_usage:.2f} MB")

# Save the best model as a .pkl file
dump(best_rf, r'D:\Project Microsoft\best_random_forest_model.pkl')
print("Model saved as best_random_forest_model.pkl")
