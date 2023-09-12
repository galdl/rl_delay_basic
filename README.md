This repository contains the implementation of the delayed-RL agent from the paper:
"Acting in Delayed Environments with Non-Stationary Markov Policies", Esther Derman<sup>\*</sup>, Gal Dalal<sup>\*</sup>, Shie Mannor (<sup>*</sup>equal contribution). 

<img src="https://github.com/galdl/rl_delay_basic/blob/master/delayed_q_diagram.png" width="600" height="330">

The agent here supports the Cartpole and Acrobot environments by OpenAI. The Atari-supported agent can be found [here](https://github.com/galdl/rl_delay_atari).

**Installation instructions:**
1. Tested with python3.7. Conda virtual env is encouraged. Other versions of python and/or environments should also be possible.
2. Clone project and cd to project dir.
3. Create virtual env:\
   Option 1 -- Tensorflow 2.2: Run `pip install -r requirements.py` (other versions of the packages in requirements.py should also be fine).\
   Option 2 -- Tensorflow 1.14: Run `conda env create -f environment. yml` to directly create a virtual env called `tf_14`.
4. To enable support of the noisy Cartpole and Acrobot experiments, modify the original gym cartpole.py and acrobot.py:\
   Option 1 -- via pip install:
      ```bash
      cd third_party
      git submodule sync && git submodule update --init --recursive
      cd gym
      git apply ../gym.patch
      pip install -e .
      ```
   Option 2 -- manually:\
      4a. Find location in site packages. E.g., "/home/username/anaconda3/envs/rl_delay_env/lib/python3.7/site-packages/gym/envs/classic_control/cartpole.py"\
      4b. Overwrite the above file with "rl_delay_basic/gym_modifications/cartpole.py". Repeat the same process for "rl_delay_basic/gym_modifications/acrobot.py".  

**Hyperparameters:**
The parameters used for the experiments in the paper are the default ones appearing in init_main.py. They are the same for all types of agents (delayed, augmented, oblivious), both noisy and non-noisy, and all delay values. The only exception is that for Cartpole epsilon_decay=0.999, while for Acrobot epsilon_decay=0.9999.

**Wandb sweep:**
Using wandb, you can easily run multiple experiments for different agents, delay values, hyperparameters, etc. An example sweep file is included the in project: example_sweep.yml. A sweep can be created via "wandb sweep example_sweep.yml", and multiple workers can be started with "wandb agent your-sweep-id". For more details see https://docs.wandb.ai/guides/sweeps/quickstart. 
  
Feel free to leave questions and raise issues. 

Happy delaying!

