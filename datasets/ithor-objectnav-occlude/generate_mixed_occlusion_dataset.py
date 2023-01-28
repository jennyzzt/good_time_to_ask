import os
import gzip
import json
from typing import List, Dict


def load_dataset(scene: str, base_directory: str) -> List[Dict]:
    # load dataset without shuffling
    filename = (
      "/".join([base_directory, scene])
      if base_directory[-1] != "/"
      else "".join([base_directory, scene])
    )
    filename += ".json.gz"
    fin = gzip.GzipFile(filename, "r")
    json_bytes = fin.read()
    fin.close()
    json_str = json_bytes.decode("utf-8")
    data = json.loads(json_str)
    return data


data_dir = os.path.join(os.getcwd(), "datasets/")


""" Combine training data """
train_out_dir = os.path.join(data_dir, f"ithor-objectnav-occlude-mixed/train/episodes/")
if not os.path.exists(train_out_dir):
    os.makedirs(train_out_dir)

kitchens = [f"FloorPlan{i}" for i in range(1, 21)]
living_rooms = [f"FloorPlan{200 + i}" for i in range(1, 21)]
bedrooms = [f"FloorPlan{300 + i}" for i in range(1, 21)]
bathrooms = [f"FloorPlan{400 + i}" for i in range(1, 21)]
train_scenes = kitchens + living_rooms + bedrooms + bathrooms

train_occ_data_dir = os.path.join(os.getcwd(), "datasets/ithor-objectnav-occlude/train/episodes")
train_data_dir = os.path.join(os.getcwd(), "datasets/ithor-objectnav/train/episodes")

for scene in train_scenes:
    occ_data = load_dataset(scene=scene, base_directory=train_occ_data_dir)
    data = load_dataset(scene=scene, base_directory=train_data_dir)
    if occ_data:
        mixed_data = occ_data + data
        # write to json.gz file
        with gzip.open(os.path.join(train_out_dir, f"{scene}.json.gz"), 'w') as f:
            f.write(json.dumps(mixed_data).encode('utf-8'))


""" Combine validation data """
val_out_dir = os.path.join(data_dir, f"ithor-objectnav-occlude-mixed/val/episodes/")
if not os.path.exists(val_out_dir):
    os.makedirs(val_out_dir)

kitchens = [f"FloorPlan{i}" for i in range(21, 26)]
living_rooms = [f"FloorPlan{200 + i}" for i in range(21, 26)]
bedrooms = [f"FloorPlan{300 + i}" for i in range(21, 26)]
bathrooms = [f"FloorPlan{400 + i}" for i in range(21, 26)]
val_scenes = kitchens + living_rooms + bedrooms + bathrooms

val_occ_data_dir = os.path.join(os.getcwd(), "datasets/ithor-objectnav-occlude/val/episodes")
val_data_dir = os.path.join(os.getcwd(), "datasets/ithor-objectnav/val/episodes")

for scene in val_scenes:
    occ_data = load_dataset(scene=scene, base_directory=val_occ_data_dir)
    data = load_dataset(scene=scene, base_directory=val_data_dir)
    if occ_data:
        mixed_data = occ_data + data
        # write to json.gz file
        with gzip.open(os.path.join(val_out_dir, f"{scene}.json.gz"), 'w') as f:
            f.write(json.dumps(mixed_data).encode('utf-8'))
