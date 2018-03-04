import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd
from numpy import genfromtxt
import pdb
from scipy import stats


eps = np.arange(0, 1e6+5e3, 5e3)

plt.rcParams['text.usetex'] = True

ddpg_1 = np.load('./results/Trust_DDPG_HalfCheetah-v1_0.0_0.0_0.npy')
ddpg_2 = np.load('./results/Trust_DDPG_HalfCheetah-v1_0.0_0.0_1.npy')
ddpg = np.mean(np.array([ddpg_1, ddpg_2]),axis=0)
ddpg_var = np.std(np.array([ddpg_1, ddpg_2]),axis=0)

l_0p001_1 = np.load('./results/Trust_DDPG_HalfCheetah-v1_0.001_0.001_0.npy')
l_0p001_2 = np.load('./results/Trust_DDPG_HalfCheetah-v1_0.001_0.001_1.npy')

l_0p001 = np.mean(np.array([l_0p001_1, l_0p001_2]),axis=0)
l_0p001_var = np.std(np.array([l_0p001_1, l_0p001_2]),axis=0)


l_0p05_1 = np.load('./results/Trust_DDPG_HalfCheetah-v1_0.05_0.05_0.npy')
l_0p05_2 = np.load('./results/Trust_DDPG_HalfCheetah-v1_0.05_0.05_1.npy')

l_0p05 = np.mean(np.array([l_0p05_1, l_0p05_2]),axis=0)
l_0p05_var = np.std(np.array([l_0p05_1, l_0p05_2]),axis=0)


l_0p1_1 = np.load('./results/Trust_DDPG_HalfCheetah-v1_0.1_0.1_0.npy')
l_0p1_2 = np.load('./results/Trust_DDPG_HalfCheetah-v1_0.1_0.1_1.npy')

l_0p1 = np.mean(np.array([l_0p1_1, l_0p1_2]),axis=0)
l_0p1_var = np.std(np.array([l_0p05_1, l_0p05_2]),axis=0)


l_0p5_1 = np.load('./results/Trust_DDPG_HalfCheetah-v1_0.5_0.5_0.npy')
l_0p5_2 = np.load('./results/Trust_DDPG_HalfCheetah-v1_0.5_0.5_1.npy')

l_0p5 = np.mean(np.array([l_0p5_1, l_0p5_2]),axis=0)
l_0p5_var = np.var(np.array([l_0p05_1, l_0p05_2]),axis=0)

l_1_1 = np.load('./results/Trust_DDPG_HalfCheetah-v1_1.0_1.0_0.npy')
l_1_2 = np.load('./results/Trust_DDPG_HalfCheetah-v1_1.0_1.0_1.npy')

l_1 = np.mean(np.array([l_1_1, l_1_2]),axis=0)
l_1_var = np.var(np.array([l_0p05_1, l_0p05_2]),axis=0)





def comparison_plot(stats1, stats2, stats3, stats4, stats5, stats6, smoothing_window=5, noshow=False):

    fig = plt.figure(figsize=(16, 8))
    ax = plt.subplot()
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Arial')
        label.set_fontsize(28)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax.xaxis.get_offset_text().set_fontsize(20)
    axis_font = {'fontname':'Arial', 'size':'32'}

    rewards_smoothed_1 = pd.Series(stats1).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_2 = pd.Series(stats2).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_3 = pd.Series(stats3).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_4 = pd.Series(stats4).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_5 = pd.Series(stats5).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_6 = pd.Series(stats6).rolling(smoothing_window, min_periods=smoothing_window).mean()

    cum_rwd_1, = plt.plot(eps, rewards_smoothed_1, color = "red", linewidth=2.5, linestyle='solid', label="Regularizer, lambda = 0.001")    
    plt.fill_between( eps, rewards_smoothed_1 + l_0p001_var,   rewards_smoothed_1 - l_0p001_var, alpha=0.2, edgecolor='red', facecolor='red')

    cum_rwd_2, = plt.plot(eps, rewards_smoothed_2, color = "blue", linewidth=2.5, linestyle='dashed', label="Regularizer, lambda = 0.05" )  
    plt.fill_between( eps, rewards_smoothed_2 + l_0p05_var,   rewards_smoothed_2 - l_0p05_var, alpha=0.2, edgecolor='blue', facecolor='blue')

    cum_rwd_3, = plt.plot(eps, rewards_smoothed_3, color = "black", linewidth=2.5, linestyle='dashdot', label="Regularizer, lambda = 0.1" )  
    plt.fill_between( eps, rewards_smoothed_3 + l_0p1_var,   rewards_smoothed_3 - l_0p1_var, alpha=0.2, edgecolor='black', facecolor='black')

    cum_rwd_4, = plt.plot(eps, rewards_smoothed_4, color = "green", linewidth=2.5, linestyle='dashdot', label="Regularizer, lambda = 0.5" )  
    plt.fill_between( eps, rewards_smoothed_4 + l_0p5_var,   rewards_smoothed_4 - l_0p5_var, alpha=0.2, edgecolor='green', facecolor='green')

    cum_rwd_5, = plt.plot(eps, rewards_smoothed_5, color = "yellow", linewidth=2.5, linestyle='dashdot', label="Regularizer, lambda = 1.0" )  
    plt.fill_between( eps, rewards_smoothed_5 + l_1_var,   rewards_smoothed_5 - l_1_var, alpha=0.2, edgecolor='yellow', facecolor='yellow')

    cum_rwd_6, = plt.plot(eps, rewards_smoothed_6, color = "magenta", linewidth=2.5, linestyle='dashdot', label="Regularizer, lambda = 0.0 (DDPG)" )  
    plt.fill_between( eps, rewards_smoothed_6 + ddpg_var,   rewards_smoothed_6 - ddpg_var, alpha=0.2, edgecolor='magenta', facecolor='magenta')


    plt.legend(handles=[cum_rwd_1, cum_rwd_2, cum_rwd_3, cum_rwd_4, cum_rwd_5, cum_rwd_6],  loc='lower right', prop={'size' : 26})
    plt.xlabel("Timesteps",**axis_font)
    plt.ylabel("Average Returns", **axis_font)
    plt.title("DDPG with HalfCheetah Environment - Critic Network Activations", **axis_font)
  
    plt.show()

    fig.savefig('ddpg_halfcheetah_value_activations.png')
    
    return fig




def plot_results(stats1, stats2, stats3, stats4, stats5, stats6, smoothing_window=5, noshow=False):

    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot()
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Arial')
        label.set_fontsize(28)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax.xaxis.get_offset_text().set_fontsize(20)
    axis_font = {'fontname':'Arial', 'size':'32'}

    rewards_smoothed_1 = pd.Series(stats1).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_2 = pd.Series(stats2).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_3 = pd.Series(stats3).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_4 = pd.Series(stats4).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_5 = pd.Series(stats5).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_6 = pd.Series(stats6).rolling(smoothing_window, min_periods=smoothing_window).mean()

    cum_rwd_1, = plt.plot(eps, rewards_smoothed_1, color = "red", linewidth=2.5, linestyle='solid', label="Regularizer, lambda = 0.001")    
    #plt.fill_between( eps, rewards_smoothed_1 + l_0p05_var,   rewards_smoothed_1 - l_0p001_var, alpha=0.2, edgecolor='red', facecolor='red')

    cum_rwd_2, = plt.plot(eps, rewards_smoothed_2, color = "blue", linewidth=2.5, linestyle='dashed', label="Regularizer, lambda = 0.05" )  
    #plt.fill_between( eps, rewards_smoothed_2 + l_0p05_var,   rewards_smoothed_2 - l_0p05_var, alpha=0.2, edgecolor='blue', facecolor='blue')

    cum_rwd_3, = plt.plot(eps, rewards_smoothed_3, color = "black", linewidth=2.5, linestyle='dashdot', label="Regularizer, lambda = 0.1" )  
    #plt.fill_between( eps, rewards_smoothed_3 + l_0p1_var,   rewards_smoothed_3 - l_0p1_var, alpha=0.2, edgecolor='black', facecolor='black')

    cum_rwd_4, = plt.plot(eps, rewards_smoothed_4, color = "green", linewidth=2.5, linestyle='dashdot', label="Regularizer, lambda = 0.5" )  
    #plt.fill_between( eps, rewards_smoothed_3 + l_0p1_var,   rewards_smoothed_3 - l_0p1_var, alpha=0.2, edgecolor='black', facecolor='black')

    cum_rwd_5, = plt.plot(eps, rewards_smoothed_5, color = "yellow", linewidth=2.5, linestyle='dashdot', label="Regularizer, lambda = 1.0" )  
    #plt.fill_between( eps, rewards_smoothed_3 + l_0p1_var,   rewards_smoothed_3 - l_0p1_var, alpha=0.2, edgecolor='black', facecolor='black')

    cum_rwd_6, = plt.plot(eps, rewards_smoothed_6, color = "cyan", linewidth=2.5, linestyle='dashdot', label="Regularizer, lambda = 0.0 (DDPG)" )  
    #plt.fill_between( eps, rewards_smoothed_3 + l_0p1_var,   rewards_smoothed_3 - l_0p1_var, alpha=0.2, edgecolor='black', facecolor='black')


    plt.legend(handles=[cum_rwd_1, cum_rwd_2, cum_rwd_3, cum_rwd_4, cum_rwd_5, cum_rwd_6],  loc='lower right', prop={'size' : 26})
    plt.xlabel("Timesteps",**axis_font)
    plt.ylabel("Average Returns", **axis_font)
    plt.title("Trust Region DDPG (with lambda values)", **axis_font)
  
    plt.show()

    fig.savefig('trust_region_ddpg.png')
    
    return fig

def main():
   #comparison_plot(l_0p001, l_0p05, l_0p1, l_0p5, l_1, ddpg)
   plot_results(l_0p001, l_0p05, l_0p1, l_0p5, l_1, ddpg)

if __name__ == '__main__':
    main()