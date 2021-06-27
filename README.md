This repository contains the implementation of the delayed-RL agent from the paper:
"Acting in Delayed Environments with Non-Stationary Markov Policies", Esther Derman<sup>\*</sup>, Gal Dalal<sup>\*</sup>, Shie Mannor (<sup>*</sup>equal contribution). 

The agent here supports the Cartpole and Acrobot environments by OpenAI. The Atari-supported agent will be released in a separate repository.

**Installation instructions:**
1. Tested with python3.7. Conda virtual env is encouraged. Other versions of python and/or environments should also be possible.
2. Clone project and cd to project dir.
3. Run "pip install -r requirements.py" (other versions of the packages in requirements.py should also be fine).
4. To enable support of the noisy Cartpole and Acrobot experiments, replace the original gym cartpole.py and acrobot.py:
  4a. Find location in site packages. E.g., "/home/username/anaconda3/envs/rl_delay_env/lib/python3.7/site-packages/gym/envs/classic_control/cartpole.py"
  4b. Overwrite the above file with "rl_delay_basic/gym_modifications/cartpole.py". Repeat the same process for "rl_delay_basic/gym_modifications/acrobot.py".
  
  
Feel free to leave questions and raise issues. 

Happy delaying!

