import numpy as np
import pandas as pd
import os 

def convert_to_homogeneous_transformations(data):
    transformations = {}
    
    for key, value in data.items():
        obj_info = value[0]
        rotation = np.array(obj_info["cam_R_m2c"]).reshape(3, 3)
        translation = np.array(obj_info["cam_t_m2c"])
        
        transformation = np.eye(4)
        transformation[:3, :3] = rotation
        transformation[:3, 3] = translation/1000
        
        transformations[key] = np.linalg.inv(transformation)
        
    return transformations

def save_transformations_to_csv(transformations, folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    for key, transformation in transformations.items():
        file_path = os.path.join(folder_path, f"{int(key):04d}.csv")
        with open(file_path, 'w') as f:
            for row in transformation:
                formatted_row = " ".join([f"{val:.17e}" for val in row])
                f.write(formatted_row + "\n")

data = {
    "0": [{"cam_R_m2c": [0.0963063, 0.99404401, 0.0510079, 0.57332098, -0.0135081, -0.81922001, -0.81365103, 0.10814, -0.57120699], "cam_t_m2c": [-105.3577515, -117.52119142, 1014.8770132], "obj_id": 1}],
    "1": [{"cam_R_m2c": [0.113309, 0.99136102, 0.0660649, 0.58543199, -0.0128929, -0.810619, -0.802764, 0.130527, -0.58183599], "cam_t_m2c": [-123.75479622, -117.18555277, 1020.10820079], "obj_id": 1}],
    "2": [{"cam_R_m2c": [0.14123, 0.98992503, -0.0101571, 0.58996803, -0.0923996, -0.802122, -0.79497898, 0.107292, -0.59707397], "cam_t_m2c": [-115.48054259, -19.45845384, 1060.18594215], "obj_id": 1}],
    "3": [{"cam_R_m2c": [0.122905, 0.99189001, 0.0323926, 0.58224899, -0.0456373, -0.81172901, -0.80366701, 0.118626, -0.58313602], "cam_t_m2c": [-109.77283244, -21.69397943, 1049.28910138], "obj_id": 1}],
    "4": [{"cam_R_m2c": [0.108976, 0.99175203, 0.0674766, 0.66973603, -0.0230918, -0.74224001, -0.73456001, 0.12607799, -0.66672802], "cam_t_m2c": [-103.59141046, -49.80357101, 1025.07614997], "obj_id": 1}],
    "5": [{"cam_R_m2c": [0.133884, 0.98939198, 0.0563724, 0.71664703, -0.0573733, -0.695072, -0.68446499, 0.133458, -0.71672601], "cam_t_m2c": [-128.30418943, 46.54086484, 1033.52572872], "obj_id": 1}]
}

transformations = convert_to_homogeneous_transformations(data)
save_transformations_to_csv(transformations, "data/tless/pose")

for key, transformation in transformations.items():
    print(f"Transformation for key {key}:")
    print(transformation)
