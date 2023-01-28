import gzip
import json
import random
import os
from typing import List, Dict
from threading import Thread

from ai2thor.controller import Controller
from ai2thor.util.metrics import path_distance, get_shortest_path_to_object


def get_occluded_objs(controller, scene):
    event = controller.reset(scene=scene)
    objs = event.metadata['objects']
    openable_receptacles_ids = [obj["objectId"] for obj in objs if obj["openable"] and obj["receptacle"]]
    occluded_objs = [obj for obj in objs
                     if obj["parentReceptacles"] and obj["parentReceptacles"][0] in openable_receptacles_ids and obj["pickupable"]]
    # openable_receptacles = [obj for obj in objs if obj["openable"] and obj["receptacle"]]
    # occluded_obj_ids = [obj["receptacleObjectIds"] for obj in openable_receptacles]
    print(f"{scene}: {len(occluded_objs)} occluded objs")
    return occluded_objs


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


def filter_data_one_object_type(data):
    # filter data to only contain one object type
    filtered_data = []
    object_type = ""
    for d in data:
        if not object_type:
            object_type = d["object_type"]
        if d["object_type"] == object_type:
            filtered_data.append(d)
        else:
            # assuming that the rest of the sorted dataset is not relevant
            break
    return filtered_data


def generate_occlusion_dataset(scene, controller, type="train"):
    controller.reset(scene=scene)
    data_dir = os.path.join(os.getcwd(), "datasets/")
    data = load_dataset(scene, os.path.join(data_dir, f"ithor-objectnav/{type}/episodes"))
    filtered_data = filter_data_one_object_type(data)
    # generate occlusion dataset
    occluded_objs = get_occluded_objs(controller=controller, scene=scene)
    occluded_objs = occluded_objs if occluded_objs else []
    occluded_data = []
    for occ_obj in occluded_objs:
        for i, d in enumerate(filtered_data):
            try:
                # for each data point, reuse the initial position and orientation
                occluded_d = dict()
                occluded_d["id"] = f"{scene}_{occ_obj['objectType']}_{i}"
                occluded_d["initial_orientation"] = d["initial_orientation"]
                occluded_d["initial_position"] = d["initial_position"]
                occluded_d["object_id"] = occ_obj["objectId"]
                occluded_d["object_type"] = occ_obj["objectType"]
                occluded_d["scene"] = d["scene"]
                occluded_d["target_position"] = occ_obj["position"]
                occluded_d["parent_receptacle"] = occ_obj["parentReceptacles"][0]
                # calculte the shortest path and length, estimated by the parent receptacle
                occluded_d["shortest_path"] = get_shortest_path_to_object(
                    controller=controller,
                    object_id=occ_obj["parentReceptacles"][0],
                    initial_position=occluded_d["initial_position"],
                    initial_rotation=occluded_d["initial_orientation"],
                )
                occluded_d["shortest_path_length"] = path_distance(occluded_d["shortest_path"])
                # append to dataset
                occluded_data.append(occluded_d)
            except:
                pass
    # write to json.gz file
    out_dir = os.path.join(data_dir, f"ithor-objectnav-occlude/{type}/episodes/")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with gzip.open(os.path.join(out_dir, f"{scene}.json.gz"), 'w') as f:
        f.write(json.dumps(occluded_data).encode('utf-8'))


# controller config should be same as experiment's
controller_args = {
    "agentMode": "default",
    "visibilityDistance": 1.0,

    # step sizes
    "gridSize": 0.25,
    "snapToGrid": True,
    "rotateStepDegrees": 90,

    # image modalities
    "renderDepthImage": False,
    "renderInstanceSegmentation": False,

    # camera properties
    "width": 300,
    "height": 300,
    "fieldOfView": 90,
}
controllers = [
    Controller(**controller_args),
    Controller(**controller_args),
    Controller(**controller_args),
]


def thread1():
    """ Generate training dataset """
    kitchens = [f"FloorPlan{i}" for i in range(1, 21)]
    living_rooms = [f"FloorPlan{200 + i}" for i in range(1, 21)]
    training_scenes = kitchens + living_rooms
    for scene in training_scenes:
        generate_occlusion_dataset(scene=scene, controller=controllers[0], type="train")
    print("Generated half training occluded dataset")
    controllers[0].stop()


def thread2():
    """ Generate training dataset """
    bedrooms = [f"FloorPlan{300 + i}" for i in range(1, 21)]
    bathrooms = [f"FloorPlan{400 + i}" for i in range(1, 21)]
    training_scenes = bedrooms + bathrooms
    for scene in training_scenes:
        generate_occlusion_dataset(scene=scene, controller=controllers[1], type="train")
    print("Generated half training occluded dataset")
    controllers[1].stop()


def thread3():
    """ Generate validation dataset """
    kitchens = [f"FloorPlan{i}" for i in range(21, 26)]
    living_rooms = [f"FloorPlan{200 + i}" for i in range(21, 26)]
    bedrooms = [f"FloorPlan{300 + i}" for i in range(21, 26)]
    bathrooms = [f"FloorPlan{400 + i}" for i in range(21, 26)]
    training_scenes = kitchens + living_rooms + bedrooms + bathrooms
    for scene in training_scenes:
        generate_occlusion_dataset(scene=scene, controller=controllers[2], type="val")
    print("Generated validation occluded dataset")
    controllers[2].stop()


threads = [
    Thread(target=thread1),
    Thread(target=thread2),
    Thread(target=thread3),
]
for t in threads:
    t.start()
