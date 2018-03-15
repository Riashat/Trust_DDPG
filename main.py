import numpy as np
import torch
import gym
import argparse
import os

import utils
import Trust_DDPG
import Trust_Ensemble_DDPG
import DDPG

from utils import (
	Logger,
	create_folder
)

# Runs policy for X episodes and returns average reward
def evaluate_policy(policy, eval_episodes=10):
	avg_reward = 0.
	for _ in range(eval_episodes):
		obs = env.reset()
		done = False
		while not done:
			action = policy.select_action(np.array(obs)).clip(env.action_space.low, env.action_space.high)
			obs, reward, done, _ = env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes

	print ("---------------------------------------")
	print ("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
	print ("---------------------------------------")
	return avg_reward


if __name__ == "__main__":
	
	args = utils.get_parser().parse_args()

	policy_name = args.policy_name
	env_name = args.env_name
	seed = args.seed
	start_timesteps = args.start_timesteps
	eval_freq = args.eval_freq
	max_timesteps = args.max_timesteps
	save_models = args.save_models
	expl_noise = args.expl_noise
	batch_size = args.batch_size
	discount = args.discount
	tau = args.tau
	lambda_critic = args.lambda_critic
	lambda_actor = args.lambda_actor

	file_name = "%s_%s_%s_%s_%s" % (policy_name, env_name, lambda_actor, lambda_critic, str(seed))

	print('POLICY: ',args.policy_name)
	# logger to record experiments
	logger = Logger(experiment_name = args.policy_name, environment_name = args.env_name, lambda_values = 'Lambda_' + str(args.lambda_actor) + '_' + str(args.lambda_critic),  folder = args.folder)
	logger.save_args(args)
	print ('Saving to', logger.save_folder)

	print ("---------------------------------------")
	print ("Settings: %s" % (file_name))
	print ("---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")
	# if not os.path.exists("./results_error_values"):
	# 	os.makedirs("./results_error_values")
	if save_models and not os.path.exists("./pytorch_models"):
		os.makedirs("./pytorch_models")


	env = gym.make(env_name)

	# Set seeds
	seed = np.random.randint(1,1000)
	env.seed(seed)
	torch.manual_seed(seed)
	np.random.seed(seed)
	# seed = np.random.seed
	# env.seed(seed)
	# torch.manual_seed(seed)

	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = int(env.action_space.high[0])

	# Initialize policy
	if policy_name == "DDPG": policy = DDPG.DDPG(state_dim, action_dim, max_action)
	elif policy_name == "Trust_DDPG": policy = Trust_DDPG.DDPG(state_dim, action_dim, max_action)
	elif policy_name == "Trust_DDPG_Adaptive": policy = Trust_DDPG.DDPG(state_dim, action_dim, max_action)
	elif policy_name == "Trust_Ensemble_DDPG": policy = Trust_Ensemble_DDPG.DDPG(state_dim, action_dim, max_action)

	replay_buffer = utils.ReplayBuffer()
	
	# Evaluate untrained policy
	evaluations = [evaluate_policy(policy)] 

	total_timesteps = 0
	timesteps_since_eval = 0
	episode_num = 0
	done = True 

	loss_critic_regularizer = np.array([])
	loss_critic_mse = np.array([])
	loss_critic = np.array([])
	loss_actor_regularizer = np.array([])
	loss_actor_original = np.array([])
	loss_actor = np.array([])


	while total_timesteps < max_timesteps:
		
		if done: 

			if total_timesteps != 0: 
				print(("Total T: %d Episode Num: %d Episode T: %d Reward: %f") % (total_timesteps, episode_num, episode_timesteps, episode_reward))
				if policy_name == "DDPG":
					policy.train(replay_buffer, episode_timesteps, batch_size, discount, tau)
				elif policy_name == "Trust_DDPG":
					lcr, lcm, lc, lar, lao, la = policy.train(replay_buffer, episode_timesteps, batch_size, discount, tau, lambda_critic, lambda_actor)
				elif policy_name == "Trust_DDPG_Adaptive":
					lcr, lcm, lc, lar, lao, la = policy.train_with_adaptive_lambda(replay_buffer, episode_timesteps, batch_size, discount, tau, lambda_critic, lambda_actor)
				elif policy_name == "Trust_Ensemble_DDPG":
					lcr, lcm, lc, lar, lao, la = policy.train(replay_buffer, episode_timesteps, batch_size, discount, tau, lambda_critic, lambda_actor)

				logger.record_data(lcr, lcm, lc, lar, lao, la)

			# Evaluate episode
			if timesteps_since_eval >= eval_freq:
				timesteps_since_eval %= eval_freq
				evaluations.append(evaluate_policy(policy))				
				logger.record_reward(evaluations)


			# Reset environment
			obs = env.reset()
			done = False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1 
		
		# Select action randomly or according to policy
		if total_timesteps < start_timesteps:
			action = env.action_space.sample()
		else:
			action = policy.select_action(np.array(obs))
			if expl_noise != 0: 
				action = (action + np.random.normal(0, expl_noise, size=env.action_space.shape[0])).clip(env.action_space.low, env.action_space.high)


		# Perform action
		new_obs, reward, done, _ = env.step(action) 
		done_bool = 0 if episode_timesteps + 1 == env._max_episode_steps else float(done)
		episode_reward += reward

		# Store data in replay buffer
		replay_buffer.add((obs, new_obs, action, reward, done_bool))

		obs = new_obs

		episode_timesteps += 1
		total_timesteps += 1
		timesteps_since_eval += 1

		
	# Final evaluation 
	evaluations.append(evaluate_policy(policy))
	logger.record_reward(evaluations)

	# if save_models: policy.save("%s" % (file_name), directory="./pytorch_models")
	# np.save("./results/%s" % (file_name), evaluations)
	# np.save("./results_error_values/" + "lcr_" + "%s" % (file_name), loss_critic_regularizer)
	# np.save("./results_error_values/" + "lcm_" + "%s" % (file_name), loss_critic_mse)
	# np.save("./results_error_values/" + "lc_" + "%s" % (file_name), loss_critic)
	# np.save("./results_error_values/" + "lar_" + "%s" % (file_name), loss_actor_regularizer)
	# np.save("./results_error_values/" + "lao_" + "%s" % (file_name), loss_actor_original)
	# np.save("./results_error_values/" + "la_" + "%s" % (file_name), loss_actor)

logger.save()
print ('DONE')
