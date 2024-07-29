# Collab program to train Sphere-Cube Logistic regression model

import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Read sphere cube data
df_sphere = pd.read_csv('/content/drive/MyDrive/PROJECTS/sphere_cube/tarin_data_sphere.csv')
df_cube = pd.read_csv('/content/drive/MyDrive/PROJECTS/sphere_cube/train_data_cube.csv')

# Combine data and delete redundant column
df = pd.concat([df_sphere, df_cube], ignore_index=True)
df.drop(['Unnamed: 0'], axis=1, inplace=True)

# Transform "object_type" column from string to boolean (1=torus, 0=sphere)
df['cube'] = df['object_type'].apply(lambda x: True if x == 'cube' else False)
df.drop(['object_type'], axis=1, inplace=True)

# Define features and split data into train/test sets
x = df.drop('cube', axis=1)  # Features without target variable (predictors)
y =  df['cube']  # Target variable
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train model
logmodel = LogisticRegression(max_iter=600000)
logmodel.fit(x_train, y_train)

# Evaluate model
y_pred = logmodel.predict(x_test)
print(classification_report(y_test, y_pred))

# Save model
model_path = '/content/drive/MyDrive/PROJECTS/sphere_cube/sphere_cube_model.sav'
pickle.dump(logmodel, open(model_path, 'wb'))
