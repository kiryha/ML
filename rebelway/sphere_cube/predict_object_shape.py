import hou
import pickle
import pandas as pd

# Load model
model_path = '$JOB/scenes/data/sphere_cube_model.sav'
logmodel = pickle.load(open(model_path, 'rb'))

# Get test object data
current_node = hou.pwd()
geo = current_node.geometry()
test_primitive = geo.prims()[0]

# Get data for current object
data = [{"points": len(geo.points()), "faces": len(geo.prims())}]

# Determine a cube
data_frame = pd.DataFrame(data)
prediction = logmodel.predict(data_frame)[0]
print(f">> This is a Cube: {prediction}")

