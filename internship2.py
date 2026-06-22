import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the Dataset
# Referencing the file "fee6c3e0-ccdb-4961-8df4-15bb4b002709" as specified
file_path = "fee6c3e0-ccdb-4961-8df4-15bb4b002709"

try:
    # Adjust encoding if your dataset contains special characters (common in movie titles/names)
    df = pd.read_csv(file_path, encoding='latin-1')
except Exception as e:
    print(f"Error loading file: {e}")
    # Fallback simulation structure if testing locally without the live file:
    # df = pd.DataFrame({'Genre': [...], 'Director': [...], 'Actor 1': [...], 'Rating': [...]})

# Display basic information
print("----_ Dataset Overview _----")
print(df.info())
print(df.head())

# 2. Data Cleaning & Preprocessing
# Standardize column names (lowercase and stripped of whitespace)
df.columns = df.columns.str.strip().str.lower()

# Identify the target column (typically 'rating')
target_col = 'rating'

if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found. Available columns: {list(df.columns)}")

# Drop rows where the target variable 'rating' is missing
df = df.dropna(subset=[target_col])

# Fill missing values for categorical features with a placeholder
categorical_cols = ['genre', 'director', 'actor 1', 'actor 2', 'actor 3']
# Dynamic fallback to check what columns exist in your specific dataset
existing_cat_cols = [col for col in categorical_cols if col in df.columns]

for col in existing_cat_cols:
    df[col] = df[col].fillna('Unknown')

# 3. Feature Engineering (Target Encoding)
# Since fields like 'director' and 'actor' have hundreds of unique values, 
# One-Hot Encoding would make the dataset too wide. We use Target Encoding instead.
encoded_features = []
for col in existing_cat_cols:
    # Calculate the mean rating for each category
    mean_rating = df.groupby(col)[target_col].mean()
    # Map it back to the dataframe as a new feature
    df[f'{col}_encoded'] = df[col].map(mean_rating)
    encoded_features.append(f'{col}_encoded')

# Include any numerical features if present (e.g., year, duration, votes)
numerical_features = [col for col in ['year', 'duration', 'votes'] if col in df.columns]
for col in numerical_features:
    # Clean numerical strings if necessary (e.g., removing 'min' from duration or commas from votes)
    if df[col].dtype == 'object':
        df[col] = df[col].astype(str).str.extract(r'(\d+)').astype(float)
    df[col] = df[col].fillna(df[col].median())

# Combine all processed features
features = encoded_features + numerical_features

X = df[features]
y = df[target_col]

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_split_size=0.2, random_state=42)

# 5. Model Selection & Training
print("\n--- Training Model ---")
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# 6. Evaluation
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation Results ---")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2) Score: {r2:.2f}")

# 7. Feature Importance Plot
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=[features[i] for i in indices], palette='viridis')
plt.title('Feature Importances in Predicting Movie Ratings')
plt.xlabel('Relative Importance')
plt.ylabel('Features')
plt.tight_layout()
plt.show()