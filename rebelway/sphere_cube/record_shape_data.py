import hou
import pandas as pd

current_node = hou.pwd()
geo = current_node.geometry()

objects = []
number_of_objects = current_node.parm("number_of_objects").eval()
object_type_index = current_node.parm("object_type_index").eval()

if object_type_index == 0:
    object_type = 'cube'
else:
    object_type = 'sphere'

csv_path = f'$JOB/scenes/data/train_data_{object_type}.csv'

processed_objects = []
data = []

for primitive in geo.prims():
    object_index = primitive.attribValue('object_index')

    if object_index not in processed_objects:
        processed_objects.append(object_index)

        data.append({
            'points': primitive.attribValue('points'),
            'faces': primitive.attribValue('faces'),
            'object_type': object_type})

data_frame = pd.DataFrame(data)
data_frame.to_csv(csv_path)
print(f">> Data Set Exported for {object_type}")
