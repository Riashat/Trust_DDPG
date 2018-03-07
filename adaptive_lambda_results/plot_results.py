import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd
from numpy import genfromtxt
import pdb
from scipy import stats


eps = np.arange(0, 1e6+5e3, 5e3)

plt.rcParams['text.usetex'] = True

ddpg = np.load('./Trust_DDPG_HalfCheetah-v1_0.0_0.0_0.npy')

l_0p001 = np.load('./Trust_DDPG_Adaptive_HalfCheetah-v1_0.001_0.001_0.npy')

l_0p005 = np.load('./Trust_DDPG_Adaptive_HalfCheetah-v1_0.005_0.005_0.npy')

l_0p01 = np.load('./Trust_DDPG_Adaptive_HalfCheetah-v1_0.01_0.01_0.npy')


l_0p05 = np.load('./Trust_DDPG_Adaptive_HalfCheetah-v1_0.05_0.05_0.npy')

l_0p075 = np.load('./Trust_DDPG_Adaptive_HalfCheetah-v1_0.075_0.075_0.npy')


l_0p1 = np.load('./Trust_DDPG_Adaptive_HalfCheetah-v1_0.1_0.1_0.npy')

l_0p25 = np.load('./Trust_DDPG_Adaptive_HalfCheetah-v1_0.25_0.25_0.npy')

l_0p5 = np.load('./Trust_DDPG_Adaptive_HalfCheetah-v1_0.5_0.5_0.npy')

l_0p75 = np.load('./Trust_DDPG_Adaptive_HalfCheetah-v1_0.75_0.75_0.npy')

l_0p9 = np.load('./Trust_DDPG_Adaptive_HalfCheetah-v1_0.9_0.9_0.npy')

l_1 = np.load('./Trust_DDPG_Adaptive_HalfCheetah-v1_1.0_1.0_0.npy')





def plot_results(stats2, stats3, stats5, stats8, stats9, smoothing_window=5, noshow=False):

    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot()
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Arial')
        label.set_fontsize(28)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax.xaxis.get_offset_text().set_fontsize(20)
    axis_font = {'fontname':'Arial', 'size':'32'}

    # rewards_smoothed_1 = pd.Series(stats1).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_2 = pd.Series(stats2).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_3 = pd.Series(stats3).rolling(smoothing_window, min_periods=smoothing_window).mean()
    # rewards_smoothed_4 = pd.Series(stats4).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_5 = pd.Series(stats5).rolling(smoothing_window, min_periods=smoothing_window).mean()
    # rewards_smoothed_6 = pd.Series(stats6).rolling(smoothing_window, min_periods=smoothing_window).mean()
    # rewards_smoothed_7 = pd.Series(stats7).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_8 = pd.Series(stats8).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_9 = pd.Series(stats9).rolling(smoothing_window, min_periods=smoothing_window).mean()
    

    # cum_rwd_1, = plt.plot(eps, rewards_smoothed_1, color = "red", linewidth=2.5, linestyle='solid', label="Regularizer, lambda = 0.001")    
    # #plt.fill_between( eps, rewards_smoothed_1 + l_0p05_var,   rewards_smoothed_1 - l_0p001_var, alpha=0.2, edgecolor='red', facecolor='red')

    cum_rwd_2, = plt.plot(eps, rewards_smoothed_2, color = "blue", linewidth=2.5, linestyle='dashed', label="Regularizer, lambda = 0.05" )  
    #plt.fill_between( eps, rewards_smoothed_2 + l_0p05_var,   rewards_smoothed_2 - l_0p05_var, alpha=0.2, edgecolor='blue', facecolor='blue')

    cum_rwd_3, = plt.plot(eps, rewards_smoothed_3, color = "black", linewidth=2.5, linestyle='dashdot', label="Regularizer, lambda = 0.01" )  
    #plt.fill_between( eps, rewards_smoothed_3 + l_0p1_var,   rewards_smoothed_3 - l_0p1_var, alpha=0.2, edgecolor='black', facecolor='black')

    # cum_rwd_4, = plt.plot(eps, rewards_smoothed_4, color = "green", linewidth=2.5, linestyle='dashdot', label="Regularizer, lambda = 0.05" )  
    # #plt.fill_between( eps, rewards_smoothed_3 + l_0p1_var,   rewards_smoothed_3 - l_0p1_var, alpha=0.2, edgecolor='black', facecolor='black')

    cum_rwd_5, = plt.plot(eps, rewards_smoothed_5, color = "yellow", linewidth=2.5, linestyle='dashdot', label="Regularizer, lambda = 0.1" )  
    #plt.fill_between( eps, rewards_smoothed_3 + l_0p1_var,   rewards_smoothed_3 - l_0p1_var, alpha=0.2, edgecolor='black', facecolor='black')

    # cum_rwd_6, = plt.plot(eps, rewards_smoothed_6, color = "cyan", linewidth=2.5, linestyle='dashdot', label="Regularizer, lambda = 0.25" )  
    # #plt.fill_between( eps, rewards_smoothed_3 + l_0p1_var,   rewards_smoothed_3 - l_0p1_var, alpha=0.2, edgecolor='black', facecolor='black')

    # cum_rwd_7, = plt.plot(eps, rewards_smoothed_7, color = "orange", linewidth=2.5, linestyle='dashdot', label="Regularizer, lambda = 0.5" )  
    # #plt.fill_between( eps, rewards_smoothed_3 + l_0p1_var,   rewards_smoothed_3 - l_0p1_var, alpha=0.2, edgecolor='black', facecolor='black')

    cum_rwd_8, = plt.plot(eps, rewards_smoothed_8, color = "magenta", linewidth=2.5, linestyle='dashdot', label="Regularizer, lambda = 0.75" )  
    #plt.fill_between( eps, rewards_smoothed_3 + l_0p1_var,   rewards_smoothed_3 - l_0p1_var, alpha=0.2, edgecolor='black', facecolor='black')

    cum_rwd_9, = plt.plot(eps, rewards_smoothed_9, color = "purple", linewidth=2.5, linestyle='dashdot', label="Regularizer, lambda = 0.0 (DDPG)" )  
    #plt.fill_between( eps, rewards_smoothed_3 + l_0p1_var,   rewards_smoothed_3 - l_0p1_var, alpha=0.2, edgecolor='black', facecolor='black')

    plt.legend(handles=[cum_rwd_2, cum_rwd_3, cum_rwd_5,  cum_rwd_8, cum_rwd_9],  loc='lower right', prop={'size' : 26})
    plt.xlabel("Timesteps",**axis_font)
    plt.ylabel("Average Returns", **axis_font)
    plt.title("Trust Region DDPG (with lambda values)", **axis_font)
  
    plt.show()

    fig.savefig('trust_region_ddpg.png')
    
    return fig

def main():

   #plot_results(l_0p001, l_0p005, l_0p01, l_0p05, l_0p1, l_0p25, l_0p5, l_0p75, ddpg)
   plot_results( l_0p005, l_0p01, l_0p5, l_0p75, ddpg)


if __name__ == '__main__':
    main()