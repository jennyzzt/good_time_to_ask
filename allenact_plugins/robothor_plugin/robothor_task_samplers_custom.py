import copy
import gzip
import json
import random
from typing import List, Optional, Union, Dict, Any, cast, Tuple

import gym

from allenact.base_abstractions.sensor import Sensor
from allenact.base_abstractions.task import TaskSampler
from allenact.utils.cache_utils import str_to_pos_for_cache
from allenact.utils.experiment_utils import set_seed, set_deterministic_cudnn
from allenact.utils.system import get_logger
from allenact_plugins.robothor_plugin.robothor_environment import RoboThorEnvironment
from allenact_plugins.robothor_plugin.robothor_tasks_custom import (
    ObjectNavTask,
    ObjectNavAskTask,
)


class ObjectNavTaskSampler(TaskSampler):
    def __init__(
        self,
        scenes: List[str],
        object_types: List[str],
        sensors: List[Sensor],
        max_steps: int,
        env_args: Dict[str, Any],
        action_space: gym.Space,
        rewards_config: Dict,
        seed: Optional[int] = None,
        deterministic_cudnn: bool = False,
        loop_dataset: bool = True,
        env_class=RoboThorEnvironment,
        task_class=ObjectNavTask,
        **kwargs,
    ) -> None:
        self.rewards_config = rewards_config
        self.env_args = env_args
        self.scenes = scenes
        self.env_class = env_class
        self.task_class = task_class
        self.object_types = object_types
        self.env: Optional[RoboThorEnvironment] = None
        self.sensors = sensors
        self.max_steps = max_steps
        self._action_space = action_space
        self.scene_counter: Optional[int] = None
        self.scene_order: Optional[List[str]] = None
        self.scene_id: Optional[int] = None
        # get the total number of tasks assigned to this process
        if loop_dataset:
            self.max_tasks = None
        else:
            self.max_tasks = 100  # arbitrary num of test episodes
        self.reset_tasks = self.max_tasks
        self.scene_index = 0
        self.episode_index = 0
        self.max_episodes = 100  # max episodes per scene

        self._last_sampled_task: Optional[self.task_class] = None

        self.seed: Optional[int] = None
        self.set_seed(seed)

        if deterministic_cudnn:
            set_deterministic_cudnn()

        self.reset()
        self.hard_reset = False

    def _create_environment(self) -> RoboThorEnvironment:
        env = self.env_class(**self.env_args)
        return env

    @property
    def __len__(self) -> Union[int, float]:
        """Length.

        # Returns

        Number of total tasks remaining that can be sampled. Can be float('inf').
        """
        return float("inf") if self.max_tasks is None else self.max_tasks

    @property
    def total_unique(self) -> Optional[Union[int, float]]:
        return self.reset_tasks

    @property
    def last_sampled_task(self):
        return self._last_sampled_task

    def close(self) -> None:
        if self.env is not None:
            self.env.stop()

    @property
    def all_observation_spaces_equal(self) -> bool:
        """Check if observation spaces equal.

        # Returns

        True if all Tasks that can be sampled by this sampler have the
            same observation space. Otherwise False.
        """
        return True

    @property
    def length(self) -> Union[int, float]:
        """Length.

        # Returns

        Number of total tasks remaining that can be sampled. Can be float('inf').
        """
        return float("inf") if self.max_tasks is None else self.max_tasks

    def next_task(self, force_advance_scene: bool = False):
        if self.max_tasks is not None and self.max_tasks <= 0:
            return None
        if self.episode_index >= self.max_episodes:
            self.scene_index = (self.scene_index + 1) % len(self.scenes)
            self.episode_index = 0
            self.hard_reset = True

        scene = self.scenes[self.scene_index]
        if self.env is None:
            self.env = self._create_environment()
        if self.hard_reset:
            self.env.controller.reset(scene=scene, renderInstanceSegmentation=True)
            self.hard_reset = False
        else:
            self.env.reset(scene_name=scene)

        object_types_in_scene = set([o["objectType"] for o in self.env.last_event.metadata["objects"]])
        for ot in random.sample(self.object_types, len(self.object_types)):
            if ot in ["Bread", "Cup"]:  # Reserved for unseen object testing
                continue
            if ot in object_types_in_scene:
                object_type = ot
                break
        # randomise object positions
        self.env.step(
            action="InitialRandomSpawn",
            randomSeed=random.randint(0, 2000),
            forceVisible=True,
            numPlacementAttempts=5,
            placeStationary=True,
        )

        object_ids = [
            o["objectId"]
            for o in self.env.last_event.metadata["objects"]
            if o["objectType"] == object_type
        ]
        self.env.set_object_filter(
            object_ids=object_ids
        )

        task_info = {
            "scene": scene,
            "object_type": object_type,
            "object_id": object_ids[0],
        }
        scene_to_agent_initial_pose = {
            "FloorPlan1": ({'x': -1.0, 'y': 0.9009996, 'z': 0.75}, 90.0),
            "FloorPlan2": ({"x": 0.0, "y": 0.9009996, "z": 2.5}, 180.0),
            "FloorPlan3": ({"x": 0.0, "y": 1.12320542, "z": 0.5}, 180.0),
            "FloorPlan4": ({"x": -1.75, "y": 0.9009996, "z": 1.0}, 180.0),
            "FloorPlan5": ({"x": -1.25, "y": 0.9009996, "z": 1.25}, 180.0),
            "FloorPlan6": ({"x": -1.75, "y": 0.9009996, "z": 1.0}, 180.0),
            "FloorPlan7": ({"x": -1.0, "y": 0.9009996, "z": 1.5}, 180.0),
            "FloorPlan8": ({"x": 0.5, "y": 0.9009996, "z": -0.25}, 180.0),
            "FloorPlan9": ({"x": 1.25, "y": 0.9009996, "z": 0.0}, 180.0),
            "FloorPlan10": ({"x": -1.25, "y": 0.9009996, "z": -2.25}, 180.0),
            "FloorPlan21": ({"x": -2.0, "y": 0.9009996, "z": -2.75}, 180.0),
            "FloorPlan22": ({"x": -0.25, "y": 0.9009996, "z": 1.0}, 180.0),
            "FloorPlan23": ({"x": -3.0, "y": 0.9009996, "z": -3.0}, 180.0),
            "FloorPlan24": ({"x": -2.0, "y": 0.9009996, "z": 3.0}, 180.0),
            "FloorPlan25": ({"x": -1.75, "y": 0.9009996, "z": 1.5}, 180.0),
        }
        task_info["initial_position"] = scene_to_agent_initial_pose[scene][0]
        task_info["initial_orientation"] = scene_to_agent_initial_pose[scene][1]
        task_info["initial_horizon"] = 0
        task_info["optimal_path"], task_info["optimal_steps"] = self.env.grid_path_from_point_to_object(
            object_id=task_info["object_id"],
            initial_position=task_info["initial_position"],
            initial_rotation={"y": task_info["initial_orientation"]},
        )

        self.episode_index += 1
        if self.max_tasks is not None:
            self.max_tasks -= 1
        if not self.env.teleport(
            pose=task_info["initial_position"],
            rotation=task_info["initial_orientation"],
            horizon=task_info["initial_horizon"],
        ):
            self.hard_reset = True
            return self.next_task()
        self._last_sampled_task = self.task_class(
            env=self.env,
            sensors=self.sensors,
            task_info=task_info,
            max_steps=self.max_steps,
            action_space=self._action_space,
            reward_configs=self.rewards_config,
        )
        return self._last_sampled_task

    def reset(self):
        self.episode_index = 0
        self.scene_index = 0
        self.max_tasks = self.reset_tasks

    def set_seed(self, seed: int):
        self.seed = seed
        if seed is not None:
            set_seed(seed)
