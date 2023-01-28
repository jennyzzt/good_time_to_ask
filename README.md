# Good Time to Ask: A Learning Framework for Asking for Help in Embodied Visual Navigation
This is the source code repository for the Good Time to Ask (GTA) paper. The experiments are tested on Ubuntu 18.04 and 20.04.

## Setup
Clone the repository with `git clone [to be inserted] && cd allenact_ask`.

Main Code Contribution:
* `projects/objectnav_ask`: this directory contains training scripts, which are used for training and evaluation.
* `allenact_plugins/robothor_plugin/`: this directory contains custom tasks and task samplers used for training and evaluation.
* `eval_scripts/`: this directory contains the evaluation script used to generate the uncertainty metrics.

You should install all necessary python packages for training. We recommend to install a conda environment and follow the instructions [here](https://allenact.org/installation/installation-allenact/#installing-a-conda-environment) provided by AllenAct.

## Training
To start a training, say you want to run an an agent with semi-present teacher, try running the following script:
```
python3 main.py projects/objectnav_ask/objectnav_ithor_rgbd_resnetgru_ddppo_asksegsemi -o outputs/ -s 12345
```
A few notes on the script:
* `-o outputs/` sets the output folder into which results and logs will be saved.
* `-s 12345` sets the random seed.
* `projects/objectnav_ask/objectnav_ithor_rgbd_resnetgru_ddppo_asksegsemi` should point to the experiment config file which you want to use for training

## Evaluation
To evaluate your trained model, first look for your checkpoint. For example, if your checkpoint is at `outputs/my_checkpoint.pt` based on your experiment above in training, you could run the following script:
```
python3 main.py projects/objectnav_ask/objectnav_ithor_rgbd_resnetgru_ddppo_asksegsemi -o outputs/ -s 12345 -c outputs/my_checkpoint.pt --eval
```
A few notes on the script:
* `-o outputs/` sets the output folder into which results and logs will be saved.
* `-c outputs/my_checkpoint.pt` specifies which checkpoint model to use.
* `--eval` marks the process as evaluation, otherwise the training will resume from the checkpoint model and continue.

## Uncertainty Metric
To generate the uncertainty metric, first look for the metrics log file. For example, if the metrics log file is at `outputs/metrics/my_metrics_file.json` based on the evaluation above, you could run the following script:
```
python3 eval_scripts/eval_uncertainty.py -f outputs.metrics/my_metrics_file.json
```
A few notes on the script:
* `-f outputs.metrics/my_metrics_file.json` specifies which evaluation log file to use to generate the uncertainty metrics.

## Acknowledgements
This codebase draws inspiration from the following codebases:
* [AllenAct](https://allenact.org/)
* [AI2-THOR](https://ai2thor.allenai.org/)
