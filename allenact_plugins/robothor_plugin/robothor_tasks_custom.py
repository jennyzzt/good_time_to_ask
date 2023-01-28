import math
from typing import Tuple, List, Dict, Any, Optional, Union, Sequence, cast

import gym
import numpy as np
import json

from allenact.base_abstractions.misc import RLStepResult
from allenact.base_abstractions.sensor import Sensor
from allenact.base_abstractions.task import Task
from allenact.utils.system import get_logger
from allenact.utils.tensor_utils import tile_images
from allenact_plugins.ithor_plugin.ithor_environment import IThorEnvironment
from allenact_plugins.robothor_plugin.robothor_constants import (
    MOVE_AHEAD,
    ROTATE_LEFT,
    ROTATE_RIGHT,
    END,
    LOOK_UP,
    LOOK_DOWN,
    ASK,
    OPEN,
    CLOSE,
)
from allenact_plugins.robothor_plugin.robothor_environment import RoboThorEnvironment


def spl_metric(
    success: bool, optimal_distance: float, travelled_distance: float
) -> Optional[float]:
    if not success:
        return 0.0
    elif optimal_distance < 0:
        return None
    elif optimal_distance == 0:
        if travelled_distance == 0:
            return 1.0
        else:
            return 0.0
    else:
        travelled_distance = max(travelled_distance, optimal_distance)
        return optimal_distance / travelled_distance


class ObjectNavTask(Task[RoboThorEnvironment]):
    _actions = (MOVE_AHEAD, ROTATE_LEFT, ROTATE_RIGHT, END, LOOK_UP, LOOK_DOWN)

    def __init__(
        self,
        env: RoboThorEnvironment,
        sensors: List[Sensor],
        task_info: Dict[str, Any],
        max_steps: int,
        reward_configs: Dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__(
            env=env, sensors=sensors, task_info=task_info, max_steps=max_steps, **kwargs
        )
        self.reward_configs = reward_configs
        self._took_end_action: bool = False
        self._success: Optional[bool] = False

        self._all_metadata_available = env.all_metadata_available

        self._rewards: List[float] = []
        self._distance_to_goal: List[float] = []
        self._metrics = None
        self.path: List = (
            []
        )  # the initial coordinate will be directly taken from the optimal path

        self.task_info["followed_path"] = [self.env.agent_state()]
        self.task_info["taken_actions"] = []
        self.task_info["action_names"] = self.class_action_names()

    @property
    def action_space(self):
        return gym.spaces.Discrete(len(self._actions))

    def reached_terminal_state(self) -> bool:
        return self._took_end_action

    @classmethod
    def class_action_names(cls, **kwargs) -> Tuple[str, ...]:
        return cls._actions

    def close(self) -> None:
        self.env.stop()

    def _step(self, action: Union[int, Sequence[int]]) -> RLStepResult:
        assert isinstance(action, int)
        action = cast(int, action)

        action_str = self.class_action_names()[action]

        self.task_info["taken_actions"].append(action_str)

        if action_str == END:
            self._took_end_action = True
            self._success = self._is_goal_in_range()
            self.last_action_success = self._success
        else:
            self.env.step({"action": action_str})
            self.last_action_success = self.env.last_action_success
            pose = self.env.agent_state()
            self.path.append({k: pose[k] for k in ["x", "y", "z"]})
            self.task_info["followed_path"].append(pose)
        step_result = RLStepResult(
            observation=self.get_observations(),
            reward=self.judge(),
            done=self.is_done(),
            info={"last_action_success": self.last_action_success, "action": action},
        )
        return step_result

    def render(self, mode: str = "rgb", *args, **kwargs) -> np.ndarray:
        assert mode in ["rgb", "depth"], "only rgb and depth rendering is implemented"
        if mode == "rgb":
            frame = self.env.current_frame.copy()
        elif mode == "depth":
            frame = self.env.current_depth.copy()
        else:
            raise NotImplementedError(f"Mode '{mode}' is not supported.")

        return frame

    def _is_goal_in_range(self) -> bool:
        return any(
            o["objectType"] == self.task_info["object_type"]
            for o in self.env.visible_objects()
        )

    def judge(self) -> float:
        """Judge the last event."""
        reward = self.reward_configs["step_penalty"]

        if self._took_end_action:
            if self._success:
                reward += self.reward_configs["goal_success_reward"]
            else:
                reward += self.reward_configs["failed_stop_reward"]
        elif self.num_steps_taken() + 1 >= self.max_steps:
            reward += self.reward_configs.get("reached_max_steps_reward", 0.0)

        self._rewards.append(float(reward))
        return float(reward)

    def get_observations(self, **kwargs) -> Any:
        obs = self.sensor_suite.get_observations(env=self.env, task=self)
        return obs

    def metrics(self) -> Dict[str, Any]:
        if not self.is_done():
            return {}

        metrics = super(ObjectNavTask, self).metrics()
        if self._all_metadata_available:
            dist2tget = self.env.distance_to_object_type(self.task_info["object_type"])

            spl = spl_metric(
                success=self._success,
                optimal_distance=self.task_info["optimal_steps"],
                travelled_distance=len(self.task_info["taken_actions"]),
            )

            metrics = {
                **metrics,
                "success": self._success,
                "total_reward": np.sum(self._rewards),
                "dist_to_target": dist2tget,
                "spl": 0 if spl is None else spl,
            }
        return metrics


class ObjectNavAskSegTask(Task[RoboThorEnvironment]):
    _actions = (MOVE_AHEAD, ROTATE_LEFT, ROTATE_RIGHT, END, LOOK_UP, LOOK_DOWN, ASK)

    def __init__(
        self,
        env: RoboThorEnvironment,
        sensors: List[Sensor],
        task_info: Dict[str, Any],
        max_steps: int,
        reward_configs: Dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__(
            env=env, sensors=sensors, task_info=task_info, max_steps=max_steps, **kwargs
        )
        self.reward_configs = reward_configs
        self._took_end_action: bool = False
        self._success: Optional[bool] = False

        self._all_metadata_available = env.all_metadata_available

        self._rewards: List[float] = []
        self._distance_to_goal: List[float] = []
        self._metrics = None
        self.path: List = (
            []
        )  # the initial coordinate will be directly taken from the optimal path
        self.travelled_distance = 0.0

        self.task_info["followed_path"] = [self.env.agent_state()]
        self.task_info["taken_actions"] = []
        self.task_info["action_names"] = self.class_action_names()
        # teacher is present 75% of the episodes
        self.task_info["teacher_present"] = np.random.rand() < 0.75

        self._took_ask_action: bool = False

    @property
    def action_space(self):
        return gym.spaces.Discrete(len(self._actions))

    def reached_terminal_state(self) -> bool:
        return self._took_end_action

    @classmethod
    def class_action_names(cls, **kwargs) -> Tuple[str, ...]:
        return cls._actions

    def close(self) -> None:
        self.env.stop()

    def _step(self, action: Union[int, Sequence[int]]) -> RLStepResult:
        assert isinstance(action, int)
        action = cast(int, action)

        action_str = self.class_action_names()[action]

        self.task_info["taken_actions"].append(action_str)
        self._took_ask_action = False

        if action_str == END:
            self._took_end_action = True
            self._success = self._is_goal_in_range()
            self.last_action_success = self._success
        elif action_str == ASK:
            self._took_ask_action = True
            self.last_action_success = True
        else:
            event = self.env.step({"action": action_str})
            self.last_action_success = self.env.last_action_success
            pose = self.env.agent_state()
            self.path.append({k: pose[k] for k in ["x", "y", "z"]})
            self.task_info["followed_path"].append(pose)
            if len(self.path) > 1:
                self.travelled_distance += IThorEnvironment.position_dist(
                    p0=self.path[-1], p1=self.path[-2], ignore_y=True
                )
        step_result = RLStepResult(
            observation=self.get_observations(),
            reward=self.judge(),
            done=self.is_done(),
            info={"last_action_success": self.last_action_success, "action": action},
        )
        return step_result

    def render(self, mode: str = "rgb", *args, **kwargs) -> np.ndarray:
        assert mode in ["rgb", "depth"], "only rgb and depth rendering is implemented"
        if mode == "rgb":
            frame = self.env.current_frame.copy()
        elif mode == "depth":
            frame = self.env.current_depth.copy()
        else:
            raise NotImplementedError(f"Mode '{mode}' is not supported.")

        return frame

    def _is_goal_in_range(self) -> bool:
        return any(
            o["objectType"] == self.task_info["object_type"]
            for o in self.env.visible_objects()
        )

    def judge(self) -> float:
        """Judge the last event."""
        reward = self.reward_configs["step_penalty"]

        if self._took_end_action:
            if self._success:
                reward += self.reward_configs["goal_success_reward"]
            else:
                reward += self.reward_configs["failed_stop_reward"]
        elif self.num_steps_taken() + 1 >= self.max_steps:
            reward += self.reward_configs.get("reached_max_steps_reward", 0.0)

        self._rewards.append(float(reward))
        return float(reward)

    def get_observations(self, **kwargs) -> Any:
        obs = self.sensor_suite.get_observations(env=self.env, task=self)
        return obs

    def metrics(self) -> Dict[str, Any]:
        if not self.is_done():
            return {}

        metrics = super(ObjectNavAskSegTask, self).metrics()
        if self._all_metadata_available:
            dist2tget = self.env.distance_to_object_type(self.task_info["object_type"])

            spl = spl_metric(
                success=self._success,
                optimal_distance=self.task_info["optimal_steps"],
                travelled_distance=len(self.task_info["taken_actions"]),
            )

            metrics = {
                **metrics,
                "success": self._success,
                "total_reward": np.sum(self._rewards),
                "dist_to_target": dist2tget,
                "spl": 0 if spl is None else spl,
            }
        return metrics
