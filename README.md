> Under Development
# Gym environments with adversarial disturbance agents

This contains the adversarial environments used in our work on Robust Adversarial Reinforcement Learning ([RARL](https://arxiv.org/abs/1703.02702)). We heavily build on OpenAI Gym.

## Getting Started

The environments are based on the MuJoCo environments wrapped by OpenAI Gym's environments ([info](https://gym.openai.com/envs#mujoco)). For more information on OpenAI Gym environments refer to the [Gym webpage](https://gym.openai.com/).

Since these environments use the OpenAI pyhton bindings for the MuJoCo environments, you'll need to install `mujoco-py` following [this](https://github.com/openai/mujoco-py).

## Example

```python
import gym
E = gym.make('InvertedPendulumAdv-v1')
current_observation = E.reset()

# Set maximum adversary force
E.update_adversary(6)

# Get a sample action
u = E.sample_action()
# u.pro corresponds to protagonist action, while u.adv corresponds to the adversary's action

# Perform action 
new_observation, reward, done, ~ = E.step(u)
```

## Contact
Lerrel Pinto -- lerrelpATcsDOTcmuDOTedu.
