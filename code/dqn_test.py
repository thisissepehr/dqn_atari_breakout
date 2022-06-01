# =================== Moving Average Code ===============

import numpy as np
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/drive')

saved_episode_reward_history = np.load('/content/drive/MyDrive/Colab Notebooks/gameai/episode_reward_history.npy')
data = np.convolve(saved_episode_reward_history, np.ones(100)/100, mode='valid')
 
# data to be plotted
x = data
 
# plotting
plt.title("Moving Average of Returns")
plt.xlabel("X axis(Moving Averages)")
plt.ylabel("Y axis(Episode Rewards Iteration)")
plt.plot(x, color ="red")
plt.show()

# =======================================================


# ============= Testing and Creating videos =============

from google.colab import drive
drive.mount('/content/drive')

from baselines.common.atari_wrappers import make_atari, wrap_deepmind 
import numpy as np
import tensorflow as tf
from tensorflow import keras
import gym

seed = 42

model = keras.models.load_model('/content/drive/MyDrive/Colab Notebooks/gameai/')

env = make_atari("BreakoutNoFrameskip-v4")
env = wrap_deepmind(env, frame_stack=True, scale=True) 
env.seed(seed)

env = gym.wrappers.Monitor(env, '/content/drive/MyDrive/Colab Notebooks/gameai/videos', video_callable=lambda episode_id: True, force=True)
                           

n_episodes = 20
returns = []

for _ in range(n_episodes):
  ret=0
    
  state = np.array(env.reset())
  done = False 
  while not done:
    # FIXME: Incomplete
    # Predict action Q-values from environment state
    state_tensor = tf.convert_to_tensor(state)
    state_tensor = tf.expand_dims(state_tensor, 0)
    state_tensor = np.array(state_tensor)
    action_probs = model.predict(state_tensor)
    # Take best action
    action = tf.argmax(action_probs[0]).numpy()

    # Apply the sampled action in our environment
    state_next, reward, done, _ = env.step(action)
    state_next = np.array(state_next)
    ret += reward
    state = state_next  

  returns.append(ret)

env.close()

print('Returns: {}'.format(returns))


#========================================================