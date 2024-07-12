"""
Week 4. 18_LOGISTIC_REGRESSION_HOUDINI
"""

import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Read torus and sphere data and combine into one data set
torus_path = '/content/drive/MyDrive/PROJECTS/rebelway_ml/torus.csv'
sphere_path = '/content/drive/MyDrive/PROJECTS/rebelway_ml/sphere.csv'
df_torus = pd.read_csv(torus_path)
df_sphere = pd.read_csv(sphere_path)

df = pd.concat([df_torus, df_sphere], ignore_index=True)
df.drop(['Unnamed: 0'], axis=1, inplace=True)

# # Option 1
# # Transform "object_type" column from string to boolean (1=torus, 0=sphere)
# object_type = pd.get_dummies(df['object_type'], drop_first=True)
# df.drop(['object_type'], axis=1, inplace=True)
# # Add as a new "torus" column
# df = pd.concat([df, object_type], axis=1)

# Option 2
# Transform "object_type" column from string to boolean (1=torus, 0=sphere)
df['torus'] = df['object_type'].apply(lambda x: True if x == 'torus' else False)
df.drop(['object_type'], axis=1, inplace=True)

# Define features and split data into train/test sets
x = df.drop('torus', axis=1)  # Features without target variable (predictors)
y = df['torus']  # target variable

# features = ['object_index', 'rows', 'columns', 'position_x', 'position_y', 'position_z']
# x = df[features]
# y = df['torus']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)

# Standardize features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Train model
logmodel = LogisticRegression(max_iter=600000)
logmodel.fit(x_train, y_train)

# Evaluate model
y_pred = logmodel.predict(x_test)
print(classification_report(y_test, y_pred))

# Save model
model_path = '/content/drive/MyDrive/PROJECTS/rebelway_ml/torus_sphere_model.sav'
pickle.dump(logmodel, open(model_path, 'wb'))