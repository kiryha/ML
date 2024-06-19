import pickle

model_file_path = "C:/Users/kko8/OneDrive/projects/rebelway/prod/3D/scenes/data/torus_sphere_model.sav"
# logmodel = pickle.load(open(model_file_path, 'rb'))

with open(model_file_path, 'rb') as pickleFile:
    model = pickle.load(pickleFile)

# Verify the model type
print(f"Model type: {type(model)}")

# Inspecting the model's attributes
if hasattr(model, 'coef_'):
    print(f"Coefficients: {model.coef_}")

if hasattr(model, 'intercept_'):
    print(f"Intercept: {model.intercept_}")

# If the model is an instance of a specific class (e.g., LinearRegression), you can access more attributes
if hasattr(model, 'score'):
    print("Model has a score method, indicating it's likely a regression model.")

# List all attributes and methods of the model
print("Model attributes and methods:")
print(dir(model))

# Detailed model information using vars (if applicable)
print("Detailed model information:")
print(vars(model))