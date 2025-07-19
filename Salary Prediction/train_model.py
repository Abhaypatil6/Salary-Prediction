import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder

# 1. Load data
data = pd.read_csv('data/employee_data.csv')

#  Drop rows where Salary is missing (target variable)
data = data.dropna(subset=['Salary'])

# 2. Handle missing values (for features)
# For numerical columns
for col in ['Age', 'Years of Experience']:
    median_value = data[col].median()
    data[col] = data[col].fillna(median_value)

# For categorical columns
for col in ['Gender', 'Education Level', 'Job Title']:
    mode_value = data[col].mode()[0]
    data[col] = data[col].fillna(mode_value)

# Check NaNs are handled
print(" Missing values after imputation:")
print(data.isnull().sum())

# 3. Features and Target
features = ["Age", "Gender", "Education Level", "Job Title", "Years of Experience"]
target = "Salary"

X = data[features]
y = data[target]

# 4. One-hot encoding for categorical columns
categorical_cols = ["Gender", "Education Level", "Job Title"]
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')  # Updated for sklearn <1.2
X_encoded = encoder.fit_transform(X[categorical_cols])

# Concatenate numerical and categorical features
other_cols = X[["Age", "Years of Experience"]].values
X_final = np.concatenate([other_cols, X_encoded], axis=1)

# 5. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# 6. Model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# 7. Save model and encoder
with open('model/salary_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('model/encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)

print(" Training complete. Model and encoder saved successfully!")
