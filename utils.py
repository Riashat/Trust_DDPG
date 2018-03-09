import numpy as np
import argparse
import numpy as np
import random
import os
import time
import json

# Code based on: 
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

# Simple replay buffer
class ReplayBuffer(object):
	def __init__(self):
		self.storage = []

	# Expects tuples of (state, next_state, action, reward, done)
	def add(self, data):
		self.storage.append(data)

	def sample(self, batch_size=100):
		ind = np.random.randint(0, len(self.storage), size=batch_size)
		x, y, u, r, d = [], [], [], [], []

		for i in ind: 
			X, Y, U, R, D = self.storage[i]
			x.append(np.array(X, copy=False))
			y.append(np.array(Y, copy=False))
			u.append(np.array(U, copy=False))
			r.append(np.array(R, copy=False))
			d.append(np.array(D, copy=False))

		return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)


def get_parser():
      parser = argparse.ArgumentParser()

      named_args = parser.add_argument_group('named arguments')

                                    
      named_args.add_argument('--policy_name', '--policy_name',
            help="""# Policy name""",
            required=False, type=str, default="Trust_DDPG")

      named_args.add_argument('--seed', '--seed',
            help="""# Sets Gym, PyTorch and Numpy seeds""",
            required=False, type=int, default=0)

      named_args.add_argument('--env_name', '--env_name',
            help="""# OpenAI gym environment name""",
            required=False, type=str, default='HalfCheetah-v1')

      named_args.add_argument('--start_timesteps', '--start_timesteps',
            help="""How many time steps purely random policy is run for""",
            required=False, type=int, default=1e4)

      named_args.add_argument('--eval_freq', '--eval_freq',
            help="""How often (time steps) we evaluate""",
            required=False, type=float, default=5e3)

      named_args.add_argument('--max_timesteps', '--max_timesteps',
            help="""# Max time steps to run environment for""",
            required=False, type=float, default=1e6)

      named_args.add_argument('--save_models', '--save_models',
            help="""# Whether or not models are saved""",
            type=bool,default=True)

      named_args.add_argument('--expl_noise', '--expl_noise',
            help="""# Std of Gaussian exploration noise""",
            required=False, type=float, default=0.1)

      named_args.add_argument('--batch_size', '--batch_size',
            help="""# Batch size for both actor and critic""",
            required=False, type=int, default=100)

      named_args.add_argument('--discount', '--discount',
            help="""# Discount factor""",
            required=False, type=float, default=0.99)

      named_args.add_argument('--tau', '--tau',
            help="""# Discount factor""",
            required=False, type=float, default=0.005)

      named_args.add_argument('--policy_noise', '--policy_noise',
            help="""# Noise added to target policy during critic update""",
            required=False, type=float, default=0.2)

      named_args.add_argument('--noise_clip', '--noise_clip',
            help="""# Range to clip target policy noise""",
            required=False, type=float, default=0.5)

      named_args.add_argument('--policy_freq', '--policy_freq',
            help="""# Frequency of delayed policy updates""",
            required=False, type=int, default=2)

      named_args.add_argument('--lambda_critic', '--lambda_critic',
            help="""# Lambda trade-off for critic regularizer""",
            required=False, type=float, default=0.1)

      named_args.add_argument('--lambda_actor', '--lambda_actor',
            help="""# Lambda trade-off for actor regularizer""",
            required=False, type=float, default=0.1)

      named_args.add_argument('-f', '--folder',
            help="""Folder to save data to""",
            required=True, type=str, default='./results/')

      return parser




create_folder = lambda f: [ os.makedirs(f) if not os.path.exists(f) else False ]
class Logger(object):
      def __init__(self, experiment_name='', lambda_values = '', folder='./results' ):
            """
            Saves experimental metrics for use later.
            :param experiment_name: name of the experiment
            :param folder: location to save data
            """
            self.rewards = []
            self.loss_critic_regularizer = []
            self.loss_critic_mse = []
            self.loss_critic = []
            self.loss_actor_regularizer = []
            self.loss_actor_original = []
            self.loss_actor = []


            self.save_folder = os.path.join(folder, experiment_name, lambda_values, time.strftime('%y-%m-%d-%H-%M-%s'))
            create_folder(self.save_folder)


      def record_reward(self, reward_return):
            self.returns_eval = reward_return

      def record_data(self, loss_critic_regularizer, loss_critic_mse, loss_critic, loss_actor_regularizer, loss_actor_original, loss_actor):
            self.loss_critic_regularizer.append(loss_critic_regularizer)
            self.loss_critic_mse.append(loss_critic_mse)
            self.loss_critic.append(loss_critic)
            self.loss_actor_regularizer.append(loss_actor_regularizer)
            self.loss_actor_original.append(loss_actor_original)
            self.loss_actor.append(loss_actor)

      def save(self):
            np.save(os.path.join(self.save_folder, "returns_eval.npy"), self.returns_eval)
            np.save(os.path.join(self.save_folder, "loss_critic_regularizer.npy"), self.loss_critic_regularizer)
            np.save(os.path.join(self.save_folder, "loss_critic_mse.npy"), self.loss_critic_mse)
            np.save(os.path.join(self.save_folder, "loss_critic.npy"), self.loss_critic)
            np.save(os.path.join(self.save_folder, "loss_actor_regularizer.npy"), self.loss_actor_regularizer)
            np.save(os.path.join(self.save_folder, "loss_actor_original.npy"), self.loss_actor_original)
            np.save(os.path.join(self.save_folder, "loss_actor.npy"), self.loss_actor)



      def save_args(self, args):
            """
            Save the command line arguments
            """
            with open(os.path.join(self.save_folder, 'params.json'), 'w') as f:
                  json.dump(dict(args._get_kwargs()), f)

