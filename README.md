
# strat

[Installation](#installation)  
[Examples](#examples)  
[Citation](#citation)  

## Installation

1. **Install the Python development environment** 

	Check if your Python environment is already configured on your system. 
	We recommend using Python 2.7, while some simulated environments also support Python 3.
	```bash
	python --version
	pip --version
	virtualenv --version
	```

	If these packages are already installed, skip to the next step.
	Otherwise, install on Ubuntu by running:
	```bash
	sudo apt update
	sudo apt install python-dev python-pip
	sudo pip install -U virtualenv  # system-wide install
	```

	Or install on mac OS by running:
	```bash
	/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
	export PATH="/usr/local/bin:/usr/local/sbin:$PATH"
	brew update
	brew install python@2  # Python 2
	sudo pip install -U virtualenv  # system-wide install
	```

2. **Create a virtual environment (recommended)** 

	Create a new virtual environment in the root directory or anywhere else:
	```bash
	virtualenv --system-site-packages -p python2.7 .venv
	```

	Activate the virtual environment every time before you use the package:
	```bash
	source .venv/bin/activate
	```

	And exit the virtual environment when you are done:
	```bash
	deactivate
	```

3. **Install the package** 

	The package can be installed by running:
	```bash
	python setup.py install
	```

4. **Download assets** 

	Download and unzip the assets folder to the root directory or anywhere else:
	```bash
	wget ftp://cs.stanford.edu/cs/cvgl/strat/assets.zip
	unzip assets.zip
	```

	If the assets folder is not in the root directory, remember to specify the 
	argument `--assets PATH_TO_ASSETS` when executing the example scripts.

## Examples

### Command Line Interface

A command line interface (CLI) is provided for debugging purposes. We recommend running the CLI to test the simulation environment after installation and data downloading: 
```bash
python tools/sawyer_cli.py --mode sim
```

Detailed usage of the CLI are currently explained in the source code of `tools/sawyer_cli.py`. Several simple functions can be test by entering the instructions below in the terminal:
* Visualize the camera images: `v`
* Mouse click and reach: `c`
* Reset the robot: `r`
* Close and open the gripper: `g` and `o`

### 4-DoF Grasping

Execute a table-top 4-DoF grasping tasks with sampled antipodal grasps:
```bash
python tools/run_env.py --env Grasp4DofEnv --policy AntipodalGrasp4DofPolicy --debug 1
```

### Planar Pushing

Execute a planar pushing tasks with a heuristic policy:
```bash
python tools/run_env.py --env PushEnv --policy HeuristicPushPolicy --debug 1
```

To execute semantic pushing tasks, we can add bindings to the configurations:
```bash
python tools/run_env.py --env PushEnv --policy HeuristicPushPolicy --env_config strat/envs/configs/push_env.yaml --policy_config strat/policies/configs/heuristic_push_policy.yaml --config_bindings "{'TASK_NAME':'crossing','LAYOUT_ID':0}" --debug 1
```

## Citation

If you find this code useful for your research, please cite:
```
@article{fang2019cavin, 
    title={Dynamics Learning with Cascaded Variational Inference for Multi-Step Manipulation},
    author={Kuan Fang and Yuke Zhu and Animesh Garg and Silvio Savarese and Li Fei-Fei}, 
    journal={Conference on Robot Learning (CoRL)}, 
    year={2019} 
}
```
