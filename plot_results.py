from stable_baselines3.common import results_plotter
import matplotlib.pyplot as plt
import numpy as np
import pickle

# #COMMENT
# Plot all reward monitors of processes using SB3
# log_dir = '/tmp/gym'
# results_plotter.plot_results(
#                 [log_dir], 4e6, results_plotter.X_TIMESTEPS, "Warehouse")
# #END COMMENT

#COMMENT 
# Plot manually stored mean rewards
with open('./log/mean_rewards.txt', 'rb') as f:
    mean_episode_rewards = pickle.load(f)

mean_episode_rewards = np.array(mean_episode_rewards)
# timesteps = HERE * mean_log_interval * processes * num_steps
timesteps = np.arange(mean_episode_rewards.shape[0]) * 10 * 16 * 10 / 1e6
print(timesteps.shape)

plt.plot(timesteps, mean_episode_rewards, color='magenta')
plt.xlabel('Timesteps[1e6]')
plt.ylabel('Mean reward')
plt.title('Warehouse with IAM')
plt.grid()
plt.show()
#END COMMENT