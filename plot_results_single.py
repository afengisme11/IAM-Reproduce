'''
Author: your name
Date: 2021-03-25 21:52:42
LastEditTime: 2021-04-01 12:57:44
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /origin/home/zheyu/Desktop/Deep_Learning/together/IAM-Reproduce/plot_results.py
'''
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
with open('./log_fa/mean_rewards_IAM.txt', 'rb') as f:
    mean_episode_rewards = pickle.load(f)

mean_episode_rewards = np.array(mean_episode_rewards)
# timesteps = HERE * mean_log_interval * processes * num_step
timesteps = np.arange(mean_episode_rewards.shape[0]) * 10 * 16 * 32/ 1e6
# print(timesteps.shape)

# EWMA
rho = 0.99 # Rho value for smoothing

s_prev = 0 # Initial value ewma value

# Empty arrays to hold the smoothed data
ewma, ewma_bias_corr = np.empty(0), np.empty(0)

for i,y in enumerate(mean_episode_rewards):
    
    # Variables to store smoothed data point
    s_cur = 0
    s_cur_bc = 0

    s_cur = rho * s_prev + (1-rho) * y
    s_cur_bc = s_cur / (1-rho**(i+1))
    
    # Append new smoothed value to array
    ewma = np.append(ewma,s_cur)
    ewma_bias_corr = np.append(ewma_bias_corr,s_cur_bc)

    s_prev = s_cur

# plt.scatter(timesteps, mean_episode_rewards, s=3) # Plot the noisy data in gray
# plt.plot(timesteps, ewma, 'r--', linewidth=3) # Plot the EWMA in red 
plt.plot(timesteps, ewma_bias_corr, 'g--', linewidth=2) # Plot the EWMA with bias correction in green
# plt.plot(timesteps, data_clean, 'orange', linewidth=3) # Plot the original data in orange
plt.xlabel('Timesteps[1e6]')
plt.ylabel('Mean reward')
plt.title('Warehouse with IAM')
plt.xlim([0,2])
# plt.ylim([26,42])
plt.grid()
plt.show()



# plt.plot(timesteps, mean_episode_rewards, color='magenta')
# plt.xlabel('Timesteps[1e6]')
# plt.ylabel('Mean reward')
# plt.title('Warehouse with IAM')
# plt.xlim([0,4])
# plt.ylim([26,42])
# plt.grid()
# plt.show()
# #END COMMENT