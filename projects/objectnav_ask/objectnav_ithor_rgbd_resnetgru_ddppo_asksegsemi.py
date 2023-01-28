import os
import glob
import platform
import gym
import numpy as np
import ai2thor
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from math import ceil
from typing import Dict, Any, List, Optional, Sequence, Tuple, cast, Union
from packaging import version
from torchvision import models

from allenact.base_abstractions.experiment_config import MachineParams, ExperimentConfig
from allenact.base_abstractions.preprocessor import Preprocessor, SensorPreprocessorGraph
from allenact.base_abstractions.sensor import SensorSuite, ExpertActionSensor, Sensor
from allenact.base_abstractions.task import TaskSampler

from allenact.algorithms.onpolicy_sync.losses.abstract_loss import AbstractActorCriticLoss
from allenact.algorithms.onpolicy_sync.losses import PPO
from allenact.algorithms.onpolicy_sync.losses.ppo import PPOConfig

from allenact.embodiedai.sensors.vision_sensors import RGBSensor, DepthSensor
from allenact.embodiedai.preprocessors.resnet import ResNetPreprocessor

from allenact_plugins.ithor_plugin.ithor_sensors import RGBSensorThor, GoalObjectTypeThorSensor
from allenact_plugins.ithor_plugin.ithor_util import horizontal_to_vertical_fov, get_open_x_displays
from allenact_plugins.ithor_plugin.ithor_environment import IThorEnvironment

from allenact_plugins.robothor_plugin.robothor_sensors import DepthSensorThor
from allenact_plugins.robothor_plugin.robothor_tasks_custom import ObjectNavAskSegTask
from allenact_plugins.robothor_plugin.robothor_task_samplers_custom import ObjectNavTaskSampler

from allenact.utils.system import get_logger
from allenact.utils.misc_utils import prepare_locals_for_super
from allenact.utils.experiment_utils import (
    Builder,
    PipelineStage,
    TrainingPipeline,
    LinearDecay,
    evenly_distribute_count_into_bins,
)

from projects.objectnav_baselines.models.object_nav_models import ResnetTensorObjectNavAskSegSemiActorCritic


class LastActionSensor(Sensor):
    def __init__(
        self,
        uuid: str = "last_action",
        **kwargs: Any,
    ):
        observation_space = gym.spaces.Discrete(len(ObjectNavAskSegTask.class_action_names()))
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
        self,
        env: IThorEnvironment,
        task: Optional[ObjectNavAskSegTask],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if len(task.task_info["taken_actions"]):
            action_str = task.task_info["taken_actions"][-1]
            action_idx = [i for i, s in enumerate(task._actions) if action_str==s][0]
            return action_idx
        else:
            return 0


class FeedbackInstanceSegSensor(Sensor):
    def __init__(
        self,
        uuid: str = "instance_segmentation",
        **kwargs: Any,
    ):
        observation_space = gym.spaces.Box(high=1, low=-1, shape=(224, 224), dtype=np.float32)
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
        self,
        env: IThorEnvironment,
        task: Optional[ObjectNavAskSegTask],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if task._took_ask_action and \
            task.task_info["object_id"] in env.last_event.instance_masks.keys() and \
                task.task_info["teacher_present"]:
            segmentation = np.float32(env.last_event.instance_masks[task.task_info["object_id"]])
        else:
            segmentation = np.zeros((224, 224), dtype=np.float32)
        segmentation = segmentation.reshape(*segmentation.shape, 1)
        return segmentation


class TeacherPresentSensor(Sensor):
    def __init__(
        self,
        uuid: str = "teacher_present",
        **kwargs: Any,
    ):
        observation_space = gym.spaces.Discrete(2)
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
        self,
        env: IThorEnvironment,
        task: Optional[ObjectNavAskSegTask],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        return task.task_info["teacher_present"]


class ObjectNaviThorRGBDPPOExperimentConfig(ExperimentConfig):
    """An Object Navigation experiment configuration in iTHOR with RGBD input."""

    TARGET_TYPES = tuple(
        sorted(
            [
                "Apple",
                "Bowl",
                "SoapBottle",
                "Potato",
                "DishSponge",
                "Cup",
                "Bread",
            ],
        )
    )

    STEP_SIZE = 0.25
    ROTATION_DEGREES = 90.0
    VISIBILITY_DISTANCE = 1.0
    STOCHASTIC = False
    HORIZONTAL_FIELD_OF_VIEW = 90

    CAMERA_WIDTH = 224
    CAMERA_HEIGHT = 224
    SCREEN_SIZE = 224
    MAX_STEPS = 500

    DEFAULT_NUM_TRAIN_PROCESSES = 16 if torch.cuda.is_available() else 1
    DEFAULT_TRAIN_GPU_IDS = tuple(range(torch.cuda.device_count()))
    DEFAULT_VALID_GPU_IDS = (torch.cuda.device_count() - 1,)
    DEFAULT_TEST_GPU_IDS = (torch.cuda.device_count() - 1,)

    TRAIN_SCENES = ["FloorPlan1", "FloorPlan2", "FloorPlan3", "FloorPlan4", "FloorPlan5", "FloorPlan6", "FloorPlan7", "FloorPlan8", "FloorPlan9", "FloorPlan10"]
    VAL_SCENES = ["FloorPlan21", "FloorPlan22", "FloorPlan23", "FloorPlan24", "FloorPlan25"]
    TEST_SCENES = VAL_SCENES

    ADVANCE_SCENE_ROLLOUT_PERIOD = None

    THOR_COMMIT_ID = "9549791ce2e7f472063a10abb1fb7664159fec23"
    THOR_IS_HEADLESS = True
    AGENT_MODE = "default"

    SENSORS = [
        RGBSensorThor(
            height=SCREEN_SIZE,
            width=SCREEN_SIZE,
            use_resnet_normalization=True,
            uuid="rgb_lowres",
        ),
        DepthSensorThor(
            height=SCREEN_SIZE,
            width=SCREEN_SIZE,
            use_normalization=True,
            uuid="depth_lowres",
        ),
        GoalObjectTypeThorSensor(object_types=TARGET_TYPES,),
        LastActionSensor(),
        FeedbackInstanceSegSensor(),
        TeacherPresentSensor(),
    ]

    def __init__(
        self,
        num_train_processes: Optional[int] = None,
        num_test_processes: Optional[int] = None,
        test_on_validation: bool = False,
        train_gpu_ids: Optional[Sequence[int]] = None,
        val_gpu_ids: Optional[Sequence[int]] = None,
        test_gpu_ids: Optional[Sequence[int]] = None,
        randomize_train_materials: bool = False,
    ):
        self.REWARD_CONFIG = {
            "step_penalty": -0.01,
            "goal_success_reward": 10.0,
            "failed_stop_reward": 0.0,
        }

        def v_or_default(v, default):
            return v if v is not None else default

        self.num_train_processes = v_or_default(
            num_train_processes, self.DEFAULT_NUM_TRAIN_PROCESSES
        )
        self.num_test_processes = v_or_default(
            num_test_processes, (5 if torch.cuda.is_available() else 1)
        )
        self.test_on_validation = test_on_validation
        self.train_gpu_ids = v_or_default(train_gpu_ids, self.DEFAULT_TRAIN_GPU_IDS)
        self.val_gpu_ids = v_or_default(val_gpu_ids, self.DEFAULT_VALID_GPU_IDS)
        self.test_gpu_ids = v_or_default(test_gpu_ids, self.DEFAULT_TEST_GPU_IDS)

        self.sampler_devices = self.train_gpu_ids
        self.randomize_train_materials = randomize_train_materials

    @classmethod
    def env_args(cls):
        assert cls.THOR_COMMIT_ID is not None

        return dict(
            width=cls.CAMERA_WIDTH,
            height=cls.CAMERA_HEIGHT,
            commit_id=cls.THOR_COMMIT_ID,
            continuousMode=False,
            applyActionNoise=cls.STOCHASTIC,
            rotateStepDegrees=cls.ROTATION_DEGREES,
            visibilityDistance=cls.VISIBILITY_DISTANCE,
            gridSize=cls.STEP_SIZE,
            snapToGrid=True,
            agentMode=cls.AGENT_MODE,
            fieldOfView=horizontal_to_vertical_fov(
                horizontal_fov_in_degrees=cls.HORIZONTAL_FIELD_OF_VIEW,
                width=cls.CAMERA_WIDTH,
                height=cls.CAMERA_HEIGHT,
            ),
            include_private_scenes=False,
            renderDepthImage=any(isinstance(s, DepthSensorThor) for s in cls.SENSORS),
            renderSegmentationImage=any(isinstance(s, FeedbackInstanceSegSensor) for s in cls.SENSORS),
        )

    @classmethod
    def tag(cls):
        return "Objectnav-asksegsemi-iTHOR-RGBD-ResNetGRU-DDPPO"

    def training_pipeline(self, **kwargs):
        # PPO
        ppo_steps = int(10000000)
        lr = 3e-4
        num_mini_batch = 1
        update_repeats = 4
        num_steps = 128
        save_interval = 500000
        log_interval = 10000 if torch.cuda.is_available() else 1
        gamma = 0.99
        use_gae = True
        gae_lambda = 0.95
        max_grad_norm = 0.5

        named_losses = {"ppo_loss": (PPO(**PPOConfig), 1.0)}

        return TrainingPipeline(
            save_interval=save_interval,
            metric_accumulate_interval=log_interval,
            optimizer_builder=Builder(optim.Adam, dict(lr=lr)),
            num_mini_batch=num_mini_batch,
            update_repeats=update_repeats,
            max_grad_norm=max_grad_norm,
            num_steps=num_steps,
            named_losses={key: val[0] for key, val in named_losses.items()},
            gamma=gamma,
            use_gae=use_gae,
            gae_lambda=gae_lambda,
            advance_scene_rollout_period=self.ADVANCE_SCENE_ROLLOUT_PERIOD,
            pipeline_stages=[
                PipelineStage(
                    loss_names=list(named_losses.keys()),
                    max_stage_steps=ppo_steps,
                    loss_weights=[val[1] for val in named_losses.values()],
                )
            ],
            lr_scheduler_builder=Builder(
                LambdaLR, {"lr_lambda": LinearDecay(steps=ppo_steps)}
            ),
        )

    def machine_params(self, mode="train", **kwargs):
        sampler_devices: Sequence[torch.device] = []
        devices: Sequence[torch.device]
        if mode == "train":
            workers_per_device = 1
            devices = (
                [torch.device("cpu")]
                if not torch.cuda.is_available()
                else cast(Tuple, self.train_gpu_ids) * workers_per_device
            )
            nprocesses = evenly_distribute_count_into_bins(
                self.num_train_processes, max(len(devices), 1)
            )
            sampler_devices = self.sampler_devices
        elif mode == "valid":
            nprocesses = 1
            devices = (
                [torch.device("cpu")]
                if not torch.cuda.is_available()
                else self.val_gpu_ids
            )
        elif mode == "test":
            devices = (
                [torch.device("cpu")]
                if not torch.cuda.is_available()
                else self.test_gpu_ids
            )
            nprocesses = evenly_distribute_count_into_bins(
                self.num_test_processes, max(len(devices), 1)
            )
        else:
            raise NotImplementedError("mode must be 'train', 'valid', or 'test'.")

        sensors = [*self.SENSORS]
        if mode != "train":
            sensors = [s for s in sensors if not isinstance(s, ExpertActionSensor)]

        sensor_preprocessor_graph = (
            SensorPreprocessorGraph(
                source_observation_spaces=SensorSuite(sensors).observation_spaces,
                preprocessors=self.preprocessors(),
            )
            if mode == "train"
            or (
                (isinstance(nprocesses, int) and nprocesses > 0)
                or (isinstance(nprocesses, Sequence) and sum(nprocesses) > 0)
            )
            else None
        )

        return MachineParams(
            nprocesses=nprocesses,
            devices=devices,
            sampler_devices=sampler_devices
            if mode == "train"
            else devices,  # ignored with > 1 gpu_ids
            sensor_preprocessor_graph=sensor_preprocessor_graph,
        )

    @classmethod
    def preprocessors(cls) -> Sequence[Union[Preprocessor, Builder[Preprocessor]]]:
        preprocessors = []

        rgb_sensor = next((s for s in cls.SENSORS if isinstance(s, RGBSensor)), None)
        if rgb_sensor is not None:
            preprocessors.append(
                ResNetPreprocessor(
                    input_height=cls.SCREEN_SIZE,
                    input_width=cls.SCREEN_SIZE,
                    output_width=7,
                    output_height=7,
                    output_dims=512,
                    pool=False,
                    torchvision_resnet_model=models.resnet18,
                    input_uuids=[rgb_sensor.uuid],
                    output_uuid="rgb_resnet",
                )
            )

        depth_sensor = next(
            (s for s in cls.SENSORS if isinstance(s, DepthSensor)), None
        )
        if depth_sensor is not None:
            preprocessors.append(
                ResNetPreprocessor(
                    input_height=cls.SCREEN_SIZE,
                    input_width=cls.SCREEN_SIZE,
                    output_width=7,
                    output_height=7,
                    output_dims=512,
                    pool=False,
                    torchvision_resnet_model=models.resnet18,
                    input_uuids=[depth_sensor.uuid],
                    output_uuid="depth_resnet",
                )
            )

        instanceseg_sensor = next(
            (s for s in cls.SENSORS if isinstance(s, FeedbackInstanceSegSensor)), None
        )
        if instanceseg_sensor is not None:
            preprocessors.append(
                ResNetPreprocessor(
                    input_height=cls.SCREEN_SIZE,
                    input_width=cls.SCREEN_SIZE,
                    output_width=7,
                    output_height=7,
                    output_dims=512,
                    pool=False,
                    torchvision_resnet_model=models.resnet18,
                    input_uuids=[instanceseg_sensor.uuid],
                    output_uuid="instanceseg_resnet",
                )
            )

        return preprocessors

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        has_rgb = any(isinstance(s, RGBSensor) for s in cls.SENSORS)
        has_depth = any(isinstance(s, DepthSensor) for s in cls.SENSORS)
        has_instanceseg = any(isinstance(s, FeedbackInstanceSegSensor) for s in cls.SENSORS)
        goal_sensor_uuid = next(
            (s.uuid for s in cls.SENSORS if isinstance(s, GoalObjectTypeThorSensor)),
            None,
        )

        return ResnetTensorObjectNavAskSegSemiActorCritic(
            action_space=gym.spaces.Discrete(len(ObjectNavAskSegTask.class_action_names())),
            observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
            goal_sensor_uuid=goal_sensor_uuid,
            rgb_resnet_preprocessor_uuid="rgb_resnet" if has_rgb else None,
            depth_resnet_preprocessor_uuid="depth_resnet" if has_depth else None,
            instanceseg_resnet_preprocessor_uuid="instanceseg_resnet" if has_instanceseg else None,
            hidden_size=512,
            goal_dims=32,
        )

    @classmethod
    def make_sampler_fn(cls, **kwargs) -> TaskSampler:
        return ObjectNavTaskSampler(**kwargs, task_class=ObjectNavAskSegTask)

    @staticmethod
    def _partition_inds(n: int, num_parts: int):
        return np.round(np.linspace(0, n, num_parts + 1, endpoint=True)).astype(
            np.int32
        )

    def _get_sampler_args_for_scene_split(
        self,
        # scenes_dir: str,
        scenes: List[str],
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]],
        seeds: Optional[List[int]],
        deterministic_cudnn: bool,
        include_expert_sensor: bool = True,
        allow_oversample: bool = False,
    ) -> Dict[str, Any]:
        # path = os.path.join(scenes_dir, "*.json.gz")
        # scenes = [scene.split("/")[-1].split(".")[0] for scene in glob.glob(path)]
        if len(scenes) == 0:
            raise RuntimeError(
                (
                    "Could find no scene dataset information in directory {}."
                    " Are you sure you've downloaded them? "
                    " If not, see https://allenact.org/installation/download-datasets/ information"
                    " on how this can be done."
                ).format(scenes_dir)
            )

        oversample_warning = (
            f"Warning: oversampling some of the scenes ({scenes}) to feed all processes ({total_processes})."
            " You can avoid this by setting a number of workers divisible by the number of scenes"
        )
        if total_processes > len(scenes):  # oversample some scenes -> bias
            if not allow_oversample:
                raise RuntimeError(
                    f"Cannot have `total_processes > len(scenes)`"
                    f" ({total_processes} > {len(scenes)}) when `allow_oversample` is `False`."
                )

            if total_processes % len(scenes) != 0:
                get_logger().warning(oversample_warning)
            scenes = scenes * int(ceil(total_processes / len(scenes)))
            scenes = scenes[: total_processes * (len(scenes) // total_processes)]
        elif len(scenes) % total_processes != 0:
            get_logger().warning(oversample_warning)

        inds = self._partition_inds(len(scenes), total_processes)

        if not self.THOR_IS_HEADLESS:
            x_display: Optional[str] = None
            if platform.system() == "Linux":
                x_displays = get_open_x_displays(throw_error_if_empty=True)

                if len([d for d in devices if d != torch.device("cpu")]) > len(
                    x_displays
                ):
                    get_logger().warning(
                        f"More GPU devices found than X-displays (devices: `{x_displays}`, x_displays: `{x_displays}`)."
                        f" This is not necessarily a bad thing but may mean that you're not using GPU memory as"
                        f" efficiently as possible. Consider following the instructions here:"
                        f" https://allenact.org/installation/installation-framework/#installation-of-ithor-ithor-plugin"
                        f" describing how to start an X-display on every GPU."
                    )
                x_display = x_displays[process_ind % len(x_displays)]

            device_dict = dict(x_display=x_display)
        else:
            device_dict = dict(gpu_device=devices[process_ind % len(devices)])

        return {
            "scenes": scenes[inds[process_ind] : inds[process_ind + 1]],
            "object_types": self.TARGET_TYPES,
            "max_steps": self.MAX_STEPS,
            "sensors": [
                s
                for s in self.SENSORS
                if (include_expert_sensor or not isinstance(s, ExpertActionSensor))
            ],
            "action_space": gym.spaces.Discrete(
                len(ObjectNavAskSegTask.class_action_names())
            ),
            "seed": seeds[process_ind] if seeds is not None else None,
            "deterministic_cudnn": deterministic_cudnn,
            "rewards_config": self.REWARD_CONFIG,
            "env_args": {**self.env_args(), **device_dict},
        }

    def train_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        res = self._get_sampler_args_for_scene_split(
            scenes=self.TRAIN_SCENES,
            # scenes_dir=os.path.join(self.TRAIN_DATASET_DIR, "episodes"),
            process_ind=process_ind,
            total_processes=total_processes,
            devices=devices,
            seeds=seeds,
            deterministic_cudnn=deterministic_cudnn,
            allow_oversample=True,
        )
        # res["scene_directory"] = self.TRAIN_DATASET_DIR
        res["loop_dataset"] = True
        # res["allow_flipping"] = True
        # res["randomize_materials_in_training"] = self.randomize_train_materials
        return res

    def valid_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        res = self._get_sampler_args_for_scene_split(
            scenes=self.VAL_SCENES,
            # scenes_dir=os.path.join(self.VAL_DATASET_DIR, "episodes"),
            process_ind=process_ind,
            total_processes=total_processes,
            devices=devices,
            seeds=seeds,
            deterministic_cudnn=deterministic_cudnn,
            include_expert_sensor=False,
            allow_oversample=False,
        )
        # res["scene_directory"] = self.VAL_DATASET_DIR
        res["loop_dataset"] = False
        return res

    def test_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        res = self._get_sampler_args_for_scene_split(
            scenes=self.TEST_SCENES,
            # scenes_dir=os.path.join(self.TEST_DATASET_DIR, "episodes"),
            process_ind=process_ind,
            total_processes=total_processes,
            devices=devices,
            seeds=seeds,
            deterministic_cudnn=deterministic_cudnn,
            include_expert_sensor=False,
            allow_oversample=False,
        )
        res["env_args"]["all_metadata_available"] = True
        res["rewards_config"] = {**res["rewards_config"]}
        # res["scene_directory"] = self.TEST_DATASET_DIR
        res["loop_dataset"] = False
        return res
