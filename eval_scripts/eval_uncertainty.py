import os
import argparse
from datetime import datetime
import json
import copy
import numpy as np
import matplotlib.pyplot as plt

from ai2thor.controller import Controller
from ai2thor.util.metrics import vector_distance

from multiprocessing import Pool, Value
from itertools import repeat


def get_curr_view_coords(controller):
  """ Get a set of 3D global coordinates from current agent's view """
  num_steps = 21  # increase this number for higher accuracy
  hit_coords = []
  for i in np.linspace(0, 1, num_steps):
    for j in np.linspace(0, 1, num_steps):
      query = controller.step(
          action="GetCoordinateFromRaycast",
          x=i,
          y=j,
      )
      coordinate = query.metadata["actionReturn"]
      hit_coords.append([*coordinate.values()])
  return np.array(hit_coords)


def in_obs(view_coords, point, threshold=0.25):
  """ Check if a given point is within threshold distance of viewed coordinates """
  point = np.array([*point.values()])
  vec_dists = view_coords - point
  norm_dists = np.linalg.norm(vec_dists, ord=2, axis=1)
  point_visibility = (norm_dists < threshold).any()
  return point_visibility


def calc_overall_uncert(uncert_mapping, num_points_offset=0):
  """ Calculate the overall uncertainty based on the uncertainty mapping """
  vals = [m[1] for m in uncert_mapping]
  # offset the number of points in the targetobj bounding box
  uncert = sum(vals) - num_points_offset
  return uncert.item()


def calc_vis_weight(dist):
  """ Calculate the weight for decreasing uncertainty based on distance from the agent 
  alpha and beta are arbitrary numbers to determine the piecewise linear shape
  """
  alpha = 1.0
  beta = 2.0
  if dist < alpha:
    return 1
  elif dist < beta:
    return 1 - (dist - alpha)/(beta - alpha)
  else:
    return 0


def in_bounding_box(bounding_box, point):
  """ Determine if a 3D point is in the 8x3 matrix bounding box 
  
  Note that this is a relaxed check
  """
  bb = np.array(bounding_box)
  min_x, min_y, min_z = bb.min(axis=0)
  max_x, max_y, max_z = bb.max(axis=0)
  return min_x <= point["x"] <= max_x and min_y <= point["y"] <= max_y and min_z <= point["z"] <= max_z


def update_uncert_mapping_nav(uncert_mapping, controller, targetobj):
  """ Update uncertainty mapping based on rgbd observation """
  new_uncert_mapping = []
  targetobj_visible = targetobj["objectId"] in controller.last_event.instance_masks.keys()
  curr_view_coords = get_curr_view_coords(controller)
  if not targetobj_visible:
    # target object is not in rgbd observation
    for (pos, uncert) in uncert_mapping:
      new_uncert = uncert
      if new_uncert == 0:
        continue
      if in_obs(curr_view_coords, pos):
        dist_from_agent = vector_distance(controller.last_event.metadata["agent"]["position"], pos)
        new_uncert -= calc_vis_weight(dist_from_agent)
      new_uncert_mapping.append((pos, max(new_uncert, 0)))
  else:
    # target object is in rgbd observation
    targetobj_vis_weight = calc_vis_weight(vector_distance(
        controller.last_event.metadata["agent"]["position"], targetobj["position"]
    ))
    targetobj_bounding_box = targetobj["objectOrientedBoundingBox"]["cornerPoints"]
    for (pos, uncert) in uncert_mapping:
      new_uncert = uncert
      if new_uncert == 0:
        continue
      if not in_bounding_box(targetobj_bounding_box, pos):
        new_uncert -= targetobj_vis_weight
        if in_obs(curr_view_coords, pos):
          dist_from_agent = vector_distance(controller.last_event.metadata["agent"]["position"], pos)
          new_uncert -= calc_vis_weight(dist_from_agent)
      new_uncert_mapping.append((pos, max(new_uncert, 0)))
  return new_uncert_mapping


def update_uncert_mapping_askseg(uncert_mapping, controller, targetobj):
  """ Update uncertainty mapping based on instance segmentation feedback """
  new_uncert_mapping = []
  curr_view_coords = get_curr_view_coords(controller)
  try:
    segmentation = np.float32(controller.last_event.instance_masks[targetobj["objectId"]])
  except:
    segmentation = np.zeros((224, 224), dtype=np.float32)
  if segmentation.any():
    # target object is identified in instance segmentation
    targetobj_bounding_box = targetobj["objectOrientedBoundingBox"]["cornerPoints"]
    targetobj_vis_weight = calc_vis_weight(vector_distance(
        controller.last_event.metadata["agent"]["position"], targetobj["position"]
    ))
    for (pos, uncert) in uncert_mapping:
      new_uncert = uncert
      if new_uncert == 0:
        continue
      if not in_bounding_box(targetobj_bounding_box, pos):
        if in_obs(curr_view_coords, pos):
          new_uncert = 0
        else:
          new_uncert -= targetobj_vis_weight
      new_uncert_mapping.append((pos, max(new_uncert, 0)))
  else:
    # target object is not identified in instance segmentation
    for (pos, uncert) in uncert_mapping:
      new_uncert = uncert
      if new_uncert == 0:
        continue
      if in_obs(curr_view_coords, pos):
        new_uncert = 0
      new_uncert_mapping.append((pos, max(new_uncert, 0)))
  return new_uncert_mapping


def render_uncertainty(controller, uncert_mapping, targetobj):
  """ Visualisation """
  fig, ax = plt.subplots(figsize=(10, 5)) 
  # plot possible target object positions
  ax.scatter(
    x=[pos["x"] for pos in poss_target_pos], 
    y=[pos["z"] for pos in poss_target_pos],
    c=[m[1] for m in uncert_mapping],
    marker='.', vmin=0.0, vmax=1.0
  )
  # plot agent position and rotation
  agent_pos = controller.last_event.metadata["agent"]["position"]
  agent_rot = controller.last_event.metadata["agent"]["rotation"]["y"]
  ax.scatter([agent_pos["x"]], [agent_pos["z"]], alpha=1.0, c="#ff7f0e", marker=(3, 0, 360-agent_rot), s=150.0)
  # plot real target object position
  ax.scatter([targetobj["position"]["x"]], [targetobj["position"]["z"]], alpha=1.0, c="#d62728", marker="X", s=150.0)
  plt.show()


def calculate_uncertainty_over_timesteps(controller, poss_target_pos, actions_taken, targetobj, render=False):
  uncert_metric = []
  ifask_uncert_metric = []

  targetobj_bounding_box = targetobj["objectOrientedBoundingBox"]["cornerPoints"]
  num_points_offset = sum([in_bounding_box(targetobj_bounding_box, pos) for pos in poss_target_pos])

  # Target object has equal possibility of being at each possible point
  uncert_mapping = [(pos, 1) for pos in poss_target_pos]
  curr_overall_uncert = calc_overall_uncert(uncert_mapping, num_points_offset)
  uncert_metric.append(curr_overall_uncert)
  if render:
    render_uncertainty(controller, uncert_mapping, targetobj)

  # Update once for timestep 0 observations
  prev_overall_uncert = curr_overall_uncert
  uncert_mapping = update_uncert_mapping_nav(uncert_mapping, controller, targetobj)
  curr_overall_uncert = calc_overall_uncert(uncert_mapping, num_points_offset)
  uncert_metric.append(curr_overall_uncert)
  if render:
    render_uncertainty(controller, uncert_mapping, targetobj)

  for action in actions_taken:
    prev_overall_uncert = curr_overall_uncert
    ifask_uncert_mapping = None

    if action in ["MoveAhead", "RotateLeft", "RotateRight", "LookUp", "LookDown"]:
      # for comparison
      ifask_uncert_mapping = update_uncert_mapping_askseg(uncert_mapping, controller, targetobj)
      # Navigation action
      controller.step(action=action)
      uncert_mapping = update_uncert_mapping_nav(uncert_mapping, controller, targetobj)
    elif action == "Ask":
      # Ask action
      uncert_mapping = update_uncert_mapping_askseg(uncert_mapping, controller, targetobj)
    else:
      # End action
      pass

    curr_overall_uncert = calc_overall_uncert(uncert_mapping, num_points_offset)
    uncert_metric.append(curr_overall_uncert)
    ifask_overall_uncert = calc_overall_uncert(ifask_uncert_mapping, num_points_offset) if ifask_uncert_mapping else curr_overall_uncert
    ifask_uncert_metric.append(ifask_overall_uncert)
    if render:
      render_uncertainty(controller, uncert_mapping, targetobj)

  return uncert_metric, ifask_uncert_metric


def init(args):
  global counter
  counter = args


def run(tasks_data, len_tasks_data):
  # for each test task
  # for task_num, task_data in enumerate(tasks_data):
  global counter
  task_data = tasks_data

  # Start controller
  THOR_COMMIT_ID = "9549791ce2e7f472063a10abb1fb7664159fec23"
  controller = Controller(commit_id=THOR_COMMIT_ID)
  controller.reset(
      agentMode="default",
      visibilityDistance=1.0,

      # step sizes
      gridSize=0.25,
      snapToGrid=True,
      rotateStepDegrees=90,

      # image modalities
      renderDepthImage=True,
      renderInstanceSegmentation=True,

      # camera properties
      width=224,
      height=224,
      fieldOfView=90
  )

  try:
    # Reset controller to task scene
    controller.reset(scene=task_data["task_info"]["scene"])

    # Get all possible target object positions
    poss_receptacle_types = ["Sink", "SinkBasin", "DiningTable", "CoffeeTable", "TVStand", "CounterTop"]
    poss_receptacle_ids = [obj["objectId"] for obj in controller.last_event.metadata["objects"] if obj["objectType"] in poss_receptacle_types]
    poss_target_pos = []
    for recep_id in poss_receptacle_ids:
      coords = controller.step(
          action="GetSpawnCoordinatesAboveReceptacle",
          objectId=recep_id,
          anywhere=True,
      ).metadata["actionReturn"]
      if coords:
        poss_target_pos.append(coords)
    poss_target_pos = [j for i in poss_target_pos for j in i]

    # filter poss_target_pos to grid ones
    filtered_poss_target_pos = []
    leeway = 0.05
    for pos in poss_target_pos:
      to_add = True
      for fpos in filtered_poss_target_pos:
        if fpos["x"] < pos["x"] + leeway and \
        fpos["x"] > pos["x"] - leeway and \
        fpos["z"] < pos["z"] + leeway and \
        fpos["z"] > pos["z"] - leeway:
          to_add = False
          break
      if to_add:
        filtered_poss_target_pos.append(pos)

    controller.step(
        action="Teleport",
        position=task_data["task_info"]["initial_position"],
        rotation=task_data["task_info"]["initial_orientation"],
        horizon=task_data["task_info"]["initial_horizon"],
        standing=True,
    )
    targetobj_id = task_data["task_info"]["object_id"]
    targetobj_type = task_data["task_info"]["object_id"].split('|')[0]
    targetobj_pos = [float(x) for x in task_data["task_info"]["object_id"].split('|')[1:]]
    targetobj_name = [obj["name"] for obj in controller.last_event.metadata["objects"] if obj["objectType"] == targetobj_type][0]
    controller.step(
      action='SetObjectPoses',
      objectPoses=[
        {
          "objectName": targetobj_name,
          "rotation": {
            "y": 0,
            "x": 0,
            "z": 0
          },
          "position": {
            "x": targetobj_pos[0],
            "y": targetobj_pos[1],
            "z": targetobj_pos[2],
          }
        },
      ]
    )
    controller.step("SetObjectFilter", objectIds=[targetobj_id], renderImage=True)

    # ready to calculate uncertainty
    uncert_metric, ifask_uncert_metric = calculate_uncertainty_over_timesteps(
      controller=controller,
      poss_target_pos=filtered_poss_target_pos,
      actions_taken=task_data["task_info"]["taken_actions"],
      targetobj=[obj for obj in controller.last_event.metadata["objects"] if obj["objectType"] == targetobj_type][0],
      render=False,
    )
    # log_data.append({
    #   "uncert": uncert_metric,
    #   "ifask_uncert": ifask_uncert_metric,
    # })
    # if task_num % 20 == 0:
    #   print(f"{datetime.now().strftime('%H:%M:%S')}: {task_num}/{len(tasks_data)} finished")
    controller.stop()
    with counter.get_lock():
      counter.value += 1
      # print(counter.value)
      if counter.value % 20 == 0:
        print(f"{datetime.now().strftime('%H:%M:%S')}: {counter.value}/{len_tasks_data} finished")
    return {
      "taken_actions": task_data["task_info"]["taken_actions"],
      "uncert": uncert_metric,
      "ifask_uncert": ifask_uncert_metric,
    }
  except:
    controller.stop()
    with counter.get_lock():
      counter.value += 1
      if counter.value % 20 == 0:
        print(f"{datetime.now().strftime('%H:%M:%S')}: {counter.value}/{len_tasks_data} finished")
    return {
      "taken_actions": task_data["task_info"]["taken_actions"],
      "uncert": "INVALID",
      "ifask_uncert": "INVALID",
    }


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-f", "--metric-file", help="Path to a metric json file", required=True, type=str)
  args = parser.parse_args()

  # load data
  with open(args.metric_file, "r") as f:
    data = json.load(f)

  # initialise log file
  log_file = os.path.join(os.path.dirname(args.metric_file), f"uncert_{os.path.basename(args.metric_file)}")
  # log_data = []

  # for each checkpoint
  chkpt_data = data[0]
  tasks_data = chkpt_data["tasks"]

  import time
  # multiprocessing
  start_time = time.time()
  num_processes = 20
  len_tasks_data = len(tasks_data)
  counter = Value('i', 0)
  pool = Pool(num_processes, initializer=init, initargs=(counter,))
  log_data = pool.starmap(run, zip(tasks_data, repeat(len_tasks_data)))
  # print(log_data)
  print("------ {} seconds taken for {} test samples ------".format(time.time() - start_time, len(log_data)))

  pool.close()

  # after all tasks, write to log file
  with open(log_file, mode="w", encoding="utf-8") as f:
    json.dump(log_data, f, indent=4)
    f.close()