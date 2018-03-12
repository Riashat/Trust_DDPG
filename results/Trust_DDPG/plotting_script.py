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


def comparison_plot(results_dict, smoothing_window=5, noshow=False):

	fig = plt.figure(figsize=(16, 8))
	ax = plt.subplot()
	for label in (ax.get_xticklabels() + ax.get_yticklabels()):
		label.set_fontname('Arial')
		label.set_fontsize(28)
	plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
	ax.xaxis.get_offset_text().set_fontsize(20)
	axis_font = {'fontname':'Arial', 'size':'32'}

	rewards_smoothed = []
	cum_rwd = []
	colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf','#999999']
	for k, key in enumerate(results_dict.keys()):
		rewards_smoothed.append(pd.Series(results_dict[key]['mean']).rolling(smoothing_window, min_periods=smoothing_window).mean())
		cum_rwd_i, = ax.plot(range(len(rewards_smoothed[k])), rewards_smoothed[k], color = colors[k%len(colors)], linewidth=2.5, linestyle='solid', label="$\lambda$ = " + str(key))
		cum_rwd.append(cum_rwd_i)
		plt.fill_between(range(len(rewards_smoothed[k])), rewards_smoothed[k] + 100+results_dict[key]['std'], rewards_smoothed[k] - 100-results_dict[key]['std'], alpha=0.2, edgecolor=colors[k%len(colors)], facecolor=colors[k%len(colors)])


	# Shrink current axis by 20%
	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

	# Put a legend to the right of the current axis
	#ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	ax.legend(handles=cum_rwd,  loc='center left', bbox_to_anchor=(1, 0.5), prop={'size' : 26})
	plt.xlabel("Timesteps",**axis_font)
	plt.ylabel("Average Returns", **axis_font)
	plt.title("DDPG with HalfCheetah Environment - Actor Only Regularization", **axis_font)
  
	plt.show()

	fig.savefig('ddpg_halfcheetah_value_activations.png')

	return fig


#Loading all data
dirs = glob.glob('*')
results = {}
for d in dirs:
	regex = re.search(r'([0-9]\.[0-9]*)_', d)
	fs = glob.glob(d+"/*/*.npy")
	if len(fs) > 0:
		print(fs)
		l = float(regex.groups()[0])
		temp_ar = []
		for f in fs:
			temp_ar.append(np.load(f)[0])
		results[l] = {}
		results[l]['mean'] = np.mean(np.array(temp_ar),axis=0)
		results[l]['std'] = np.std(np.array(temp_ar),axis=0)

comparison_plot(results)
