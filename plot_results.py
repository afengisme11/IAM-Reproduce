'''
Author: your name
Date: 2021-03-25 21:52:42
LastEditTime: 2021-04-01 23:51:39
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
with open('./log/mean_rewards_GRU.txt', 'rb') as f:
    mean_episode_rewards_GRU = pickle.load(f)
with open('./log/mean_rewards_IAM.txt', 'rb') as f:
    mean_episode_rewards_IAM = pickle.load(f)
with open('./log/mean_rewards_FNN8.txt', 'rb') as f:
    mean_episode_rewards_FNN8 = pickle.load(f)
with open('./log/mean_rewards_FNN1.txt', 'rb') as f:
    mean_episode_rewards_FNN1 = pickle.load(f)

mean_episode_rewards_GRU = np.array(mean_episode_rewards_GRU)
mean_episode_rewards_IAM = np.array(mean_episode_rewards_IAM)
mean_episode_rewards_FNN8 = np.array(mean_episode_rewards_FNN8)
mean_episode_rewards_FNN1 = np.array(mean_episode_rewards_FNN1)
# timesteps = HERE * mean_log_interval * processes * num_steps
timesteps = np.arange(mean_episode_rewards_GRU.shape[0]) * 10 * 16 * 8/ 1e6
# print(timesteps.shape)

fnn8 = np.zeros_like(timesteps) + np.mean(mean_episode_rewards_FNN8[-30:])
fnn1 = np.zeros_like(timesteps) + np.mean(mean_episode_rewards_FNN1[-30:])
# EWMA
rho = 0.99 # Rho value for smoothing

s_prev = 0 # Initial value ewma value

# Empty arrays to hold the smoothed data
ewma_GRU, ewma_bias_corr_GRU = np.empty(0), np.empty(0)
ewma_IAM, ewma_bias_corr_IAM = np.empty(0), np.empty(0)

for i,y in enumerate(mean_episode_rewards_IAM):
    
    # Variables to store smoothed data point
    s_cur = 0
    s_cur_bc = 0

    s_cur = rho * s_prev + (1-rho) * y
    s_cur_bc = s_cur / (1-rho**(i+1))
    
    # Append new smoothed value to array
    ewma_IAM = np.append(ewma_IAM,s_cur)
    ewma_bias_corr_IAM = np.append(ewma_bias_corr_IAM,s_cur_bc)

    s_prev = s_cur

rho = 0.99 # Rho value for smoothing

s_prev = 0 # Initial value ewma value
for i,y in enumerate(mean_episode_rewards_GRU):
    
    # Variables to store smoothed data point
    s_cur = 0
    s_cur_bc = 0

    s_cur = rho * s_prev + (1-rho) * y
    s_cur_bc = s_cur / (1-rho**(i+1))
    
    # Append new smoothed value to array
    ewma_GRU = np.append(ewma_GRU,s_cur)
    ewma_bias_corr_GRU = np.append(ewma_bias_corr_GRU,s_cur_bc)

    s_prev = s_cur
# plt.scatter(timesteps, mean_episode_rewards, s=3) # Plot the noisy data in gray
# plt.plot(timesteps, ewma, 'r--', linewidth=3) # Plot the EWMA in red 
plt.plot(timesteps, ewma_bias_corr_IAM, 'b--', linewidth=2, label='IAM') 
plt.plot(timesteps, ewma_bias_corr_GRU, 'y--', linewidth=2, label='GRU')
plt.plot(timesteps, fnn8, 'r--', linewidth=2, label='FNN(8 obs)')
plt.plot(timesteps, fnn1, 'k--', linewidth=2, label='FNN(1 obs)')
# plt.plot(timesteps, data_clean, 'orange', linewidth=3) # Plot the original data in orange
plt.xlabel('Timesteps[1e6]')
plt.ylabel('Mean reward')
plt.title('Warehouse with GRU')
plt.xlim([0,4])
plt.ylim([26,42])
plt.grid()
plt.legend(loc="lower right")
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