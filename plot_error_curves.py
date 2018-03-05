import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd
from numpy import genfromtxt
import pdb
from scipy import stats


eps = np.arange(0, 999, 1)

plt.rcParams['text.usetex'] = True



la_0p01 = np.load('./results_error_values/la_Trust_DDPG_HalfCheetah-v1_0.01_0.01_0.npy')
la_ddpg = np.load('./results_error_values/la_Trust_DDPG_HalfCheetah-v1_0.0_0.0_0.npy')

lao_0p01 = np.load('./results_error_values/lao_Trust_DDPG_HalfCheetah-v1_0.01_0.01_0.npy')
lao_ddpg = np.load('./results_error_values/lao_Trust_DDPG_HalfCheetah-v1_0.0_0.0_0.npy')

lar_0p01 = np.load('./results_error_values/lar_Trust_DDPG_HalfCheetah-v1_0.01_0.01_0.npy')
lar_ddpg = np.load('./results_error_values/lar_Trust_DDPG_HalfCheetah-v1_0.0_0.0_0.npy')




lc_0p01 = np.load('./results_error_values/lc_Trust_DDPG_HalfCheetah-v1_0.01_0.01_0.npy')
lc_ddpg = np.load('./results_error_values/lc_Trust_DDPG_HalfCheetah-v1_0.0_0.0_0.npy')

lcm_0p01 = np.load('./results_error_values/lcm_Trust_DDPG_HalfCheetah-v1_0.01_0.01_0.npy')
lcm_ddpg = np.load('./results_error_values/lcm_Trust_DDPG_HalfCheetah-v1_0.0_0.0_0.npy')

lcr_0p01 = np.load('./results_error_values/lcr_Trust_DDPG_HalfCheetah-v1_0.01_0.01_0.npy')
lcr_ddpg = np.load('./results_error_values/lcr_Trust_DDPG_HalfCheetah-v1_0.0_0.0_0.npy')




def actor_loss(stats5, stats6, smoothing_window=5, noshow=False):

    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot()
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Arial')
        label.set_fontsize(28)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax.xaxis.get_offset_text().set_fontsize(20)
    axis_font = {'fontname':'Arial', 'size':'32'}

    #rewards_smoothed_1 = pd.Series(stats1).rolling(smoothing_window, min_periods=smoothing_window).mean()
    #rewards_smoothed_2 = pd.Series(stats2).rolling(smoothing_window, min_periods=smoothing_window).mean()
    #rewards_smoothed_3 = pd.Series(stats3).rolling(smoothing_window, min_periods=smoothing_window).mean()
    #rewards_smoothed_4 = pd.Series(stats4).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_5 = pd.Series(stats5).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_6 = pd.Series(stats6).rolling(smoothing_window, min_periods=smoothing_window).mean()

    #cum_rwd_1, = plt.plot(eps, rewards_smoothed_1, color = "blue", linewidth=1.5, linestyle='solid', label="Actor Total Loss, lambda = 0.01")    
    #cum_rwd_2, = plt.plot(eps, rewards_smoothed_2, color = "green", linewidth=1.5, linestyle='dashed', label="Actor Total Loss, DDPG" )  
    #cum_rwd_3, = plt.plot(eps, rewards_smoothed_3, color = "blue", linewidth=1.5, linestyle='solid', label="Actor Regularizer Loss, lambda = 0.01" )  
    #cum_rwd_4, = plt.plot(eps, rewards_smoothed_4, color = "green", linewidth=1.5, linestyle='dashdot', label="Actor Regularizer Loss, DDPG" )  
    cum_rwd_5, = plt.plot(eps, rewards_smoothed_5, color = "red", linewidth=1.5, linestyle='solid', label="Actor Critic Loss, lambda = 0.01" )  
    cum_rwd_6, = plt.plot(eps, rewards_smoothed_6, color = "black", linewidth=1.5, linestyle='dashdot', label="Actor Critic Loss, DDPG" )  


    plt.legend(handles=[cum_rwd_5, cum_rwd_6],  loc='top right', prop={'size' : 26})
    plt.xlabel("Iterations",**axis_font)
    plt.ylabel("Loss", **axis_font)
    plt.title("Actor - Critic Loss Values", **axis_font)
  
    plt.show()

    fig.savefig('trust_region_ddpg.png')
    
    return fig



def critic_loss(stats3, stats4, smoothing_window=10, noshow=False):

    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot()
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Arial')
        label.set_fontsize(28)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax.xaxis.get_offset_text().set_fontsize(20)
    axis_font = {'fontname':'Arial', 'size':'32'}

    #rewards_smoothed_1 = pd.Series(stats1).rolling(smoothing_window, min_periods=smoothing_window).mean()
    #rewards_smoothed_2 = pd.Series(stats2).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_3 = pd.Series(stats3).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_4 = pd.Series(stats4).rolling(smoothing_window, min_periods=smoothing_window).mean()
    # rewards_smoothed_5 = pd.Series(stats5).rolling(smoothing_window, min_periods=smoothing_window).mean()
    # rewards_smoothed_6 = pd.Series(stats6).rolling(smoothing_window, min_periods=smoothing_window).mean()

    #cum_rwd_1, = plt.plot(eps, rewards_smoothed_1, color = "blue", linewidth=1.5, linestyle='solid', label="Critic Total Loss, lambda = 0.01")    
    #cum_rwd_2, = plt.plot(eps, rewards_smoothed_2, color = "green", linewidth=1.5, linestyle='dashed', label="Critic Total Loss, DDPG" )  
    cum_rwd_3, = plt.plot(eps, rewards_smoothed_3, color = "blue", linewidth=1.5, linestyle='solid', label="Critic Regularizer Loss, lambda = 0.01" )  
    cum_rwd_4, = plt.plot(eps, rewards_smoothed_4, color = "green", linewidth=1.5, linestyle='dashdot', label="Critic Regularizer Loss, DDPG" )  
    # cum_rwd_5, = plt.plot(eps, rewards_smoothed_5, color = "red", linewidth=1.5, linestyle='solid', label="Critic MSE Loss, lambda = 0.01" )  
    # cum_rwd_6, = plt.plot(eps, rewards_smoothed_6, color = "black", linewidth=1.5, linestyle='dashdot', label="Critic MSE Loss, DDPG" )  


    plt.legend(handles=[cum_rwd_3, cum_rwd_4],  loc='top right', prop={'size' : 26})
    plt.xlabel("Iterations",**axis_font)
    plt.ylabel("Loss", **axis_font)
    plt.title("Critic Regularizer Loss Values", **axis_font)
  
    plt.show()

    fig.savefig('trust_region_ddpg.png')
    
    return fig





def main():
    critic_loss(lcr_0p01, lcr_ddpg)

if __name__ == '__main__':
    main()