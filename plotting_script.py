import glob
import re
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd
from numpy import genfromtxt
import pdb
from scipy import stats

#eps = np.arange(0, 1e6+5e3, 5e3)
plt.rcParams['text.usetex'] = True


def comparison_plot(results_dict, smoothing_window=20, noshow=False):

	fig = plt.figure(figsize=(16, 8))
	ax = plt.subplot()
	for label in (ax.get_xticklabels() + ax.get_yticklabels()):
		label.set_fontname('Arial')
		label.set_fontsize(28)
	plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
	ax.xaxis.get_offset_text().set_fontsize(20)
	axis_font = {'fontname':'Arial', 'size':'32'}

	rewards_smoothed = []
	rewards_max = []
	rewards_min = []
	cum_rwd = []
	colors = ['#e6194b', '#3cb44b','#ffe119','#0082c8','#f58231','#911eb4','#46f0f0','#f032e6', '#d2f53c','#fabebe','#008080','#e6beff','#aa6e28','#fffac8','#800000','#aaffc3','#808000','#ffd8b1','#000080','#808080','#FFFFFF','#000000']
	#colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf','#999999', '#000000', ]
	index = 0
	for k, key in enumerate(results_dict.keys()):

		#for i, data in enumerate(results_dict[key]['all']):
		#	rewards_smoothed.append(pd.Series(data).rolling(smoothing_window, min_periods=smoothing_window).mean())
		#	cum_rwd_i, = ax.plot(range(len(rewards_smoothed[index])), rewards_smoothed[index], color = colors[k%len(colors)], linewidth=2.5, linestyle='solid', label=str(i)+": $\lambda$ = " + str(key))
		#	cum_rwd.append(cum_rwd_i)
		#	index += 1		 
		rewards_smoothed.append(pd.Series(results_dict[key]['mean']).rolling(smoothing_window, min_periods=smoothing_window).mean())
		cum_rwd_i, = ax.plot(range(len(rewards_smoothed[k])), rewards_smoothed[k], color = colors[k%len(colors)], linewidth=2.5, linestyle='solid', label="$\lambda$ = " + str(key))
		cum_rwd.append(cum_rwd_i)
		plt.fill_between(range(len(rewards_smoothed[k])), rewards_smoothed[k] + results_dict[key]['std'], rewards_smoothed[k] - results_dict[key]['std'], alpha=0.2, edgecolor=colors[k%len(colors)], facecolor=colors[k%len(colors)])
		#rewards_min.append(pd.Series(results_dict[key]['min']).rolling(smoothing_window, min_periods=smoothing_window).mean())		
		#cum_rwd_i, = ax.plot(range(len(rewards_min[k])), rewards_min[k], color = colors[k%len(colors)], linewidth=2.5, linestyle='solid', label="Min:$\lambda$ = " + str(key))
		#cum_rwd.append(cum_rwd_i)
		#rewards_max.append(pd.Series(results_dict[key]['max']).rolling(smoothing_window, min_periods=smoothing_window).mean())
		#cum_rwd_i, = ax.plot(range(len(rewards_max[k])), rewards_max[k], color = colors[k%len(colors)], linewidth=2.5, linestyle='solid', label="Max:$\lambda$ = " + str(key))
		#cum_rwd.append(cum_rwd_i)
		#plt.fill_between(range(len(rewards_max[k])), rewards_max[k], rewards_min[k], alpha=0.2, edgecolor=colors[k%len(colors)], facecolor=colors[k%len(colors)])

	# Shrink current axis by 20%
	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

	# Put a legend to the right of the current axis
	#ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	ax.legend(handles=cum_rwd,  loc='center left', bbox_to_anchor=(1, 0.5), prop={'size' : 26})
	plt.xlabel("Episodes (1e6 timesteps)",**axis_font)
	plt.ylabel("Average Returns", **axis_font)
	plt.title("DDPG with Walker Environment - Actor and Critic Regularization", **axis_font)
  
	# plt.show()

	plt.savefig('./results_analysis/ddpg_walker_best_comparisons.png')

	return fig


#Loading all data
dirs = glob.glob('./all_results/Trust_DDPG/Walker2d-v1/*')
results = {}
for d in dirs:
	regex = re.search(r'([0-9]*\.[0-9]*)_([0-9]*\.[0-9]*)', d)
	fs = glob.glob(d+"/*/returns_eval.npy")
	if len(fs) > 0:
		l = (float(regex.groups()[0]), float(regex.groups()[1]))
		print(l, len(fs[:5]))
		temp_ar = []
		for f in fs[:5]:
			temp_ar.append(np.load(f))
		#if l in [(0.0, 0.0), (75.0, 50.0), (75.0, 30.0), (75.0, 20.0)]: #best l_actor, l_critic
		#if l in [(0.0, 0.0), ()]
		if l in [(0.0, 0.0), (125.0, 0.0), (75.0, 30.0), (75.0, 20.0)]:

			results[l] = {}
			results[l]['mean'] = np.mean(np.array(temp_ar),axis=0)
			results[l]['std'] = np.std(np.array(temp_ar),axis=0)
			results[l]['max'] = np.max(np.array(temp_ar),axis=0)
			results[l]['min'] = np.min(np.array(temp_ar),axis=0)
			results[l]['all'] = np.array(temp_ar)

comparison_plot(results)
