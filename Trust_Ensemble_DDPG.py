import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math

import utils

## Here we use an Ensemble of Critics 
## and use the gradient information from this ensemble of critics
## for training the actor.

## Much like repeated critic training, for training the actor. 


def var(tensor, volatile=False):
	if torch.cuda.is_available():
		return Variable(tensor, volatile=volatile).cuda()
	else:
		return Variable(tensor, volatile=volatile)


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 400)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, action_dim)
		
		self.max_action = max_action

	
	def forward(self, x):
		x = F.relu(self.l1(x))
		x = F.relu(self.l2(x))
		x = self.max_action * F.tanh(self.l3(x)) 
		return x 


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		self.l1 = nn.Linear(state_dim, 400)
		self.l2 = nn.Linear(400 + action_dim, 300)
		self.l3 = nn.Linear(300, 1)
		self.d1 = nn.Dropout(p=0.2)
		self.d2 = nn.Dropout(p=0.2)


	def forward(self, x, u, use_dropout=False):
		if use_dropout:
			x = self.d1(F.relu(self.l1(x)))
			x = self.d2(F.relu(self.l2(torch.cat([x, u], 1))))
			x = self.l3(x)

		else:
			x = F.relu(self.l1(x))
			x = F.relu(self.l2(torch.cat([x, u], 1)))
			x = self.l3(x)

		return x 


class DDPG(object):
	def __init__(self, state_dim, action_dim, max_action):
		self.actor = Actor(state_dim, action_dim, max_action)
		self.actor_target = Actor(state_dim, action_dim, max_action)
		self.actor_target.load_state_dict(self.actor.state_dict())
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

		self.critic = Critic(state_dim, action_dim)
		self.critic_target = Critic(state_dim, action_dim)
		self.critic_target.load_state_dict(self.critic.state_dict())
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), weight_decay=1e-2)		

		if torch.cuda.is_available():
			self.actor = self.actor.cuda()
			self.actor_target = self.actor_target.cuda()
			self.critic = self.critic.cuda()
			self.critic_target = self.critic_target.cuda()

		self.criterion = nn.MSELoss()
		self.state_dim = state_dim


	def select_action(self, state):
		state = var(torch.FloatTensor(state.reshape(-1, self.state_dim)), volatile=True)
		return self.actor(state).cpu().data.numpy().flatten()

	def critic_retrain(self, replay_buffer, iterations, batch_size, discount, tau, lambda_critic, retrain_n=5):
    		
			for r in range(retrain_n):
				x, y, u, r, d = replay_buffer.sample(batch_size)
				state = var(torch.FloatTensor(x))
				action = var(torch.FloatTensor(u))
				next_state = var(torch.FloatTensor(y), volatile=True)
				done = var(torch.FloatTensor(1 - d))
				reward = var(torch.FloatTensor(r))

				target_Q = self.critic_target(next_state, self.actor_target(next_state))
				target_Q = reward + (done * discount * target_Q)
				target_Q.volatile = False 				

				current_Q = self.critic(state, action, use_dropout=True)
				critic_mse = self.criterion(current_Q, target_Q)

				target_Q_current_state = self.critic_target(state, self.actor(state))
				target_Q_current_state = Variable(target_Q_current_state.data)
				target_Q_current_state.volatile = False

				critic_regularizer = self.criterion(current_Q, target_Q_current_state)
				critic_loss = critic_mse + lambda_critic * critic_regularizer
	
				self.critic_optimizer.zero_grad()
				critic_loss.backward()
				self.critic_optimizer.step()

				loss_critic_regularizer = critic_regularizer.data.cpu().numpy()
				loss_critic_mse = critic_mse.data.cpu().numpy()
				loss_critic = critic_loss.data.cpu().numpy()

			return loss_critic_regularizer, loss_critic_mse, loss_critic


	def actor_update(self, replay_buffer, iterations, batch_size, discount, tau, lambda_actor):
    		
		x, y, u, r, d = replay_buffer.sample(batch_size)
		state = var(torch.FloatTensor(x))
		action = var(torch.FloatTensor(u))
		next_state = var(torch.FloatTensor(y), volatile=True)
		done = var(torch.FloatTensor(1 - d))
		reward = var(torch.FloatTensor(r))

		target_actor = self.actor_target(state)
		target_actor = Variable(target_actor.data)
		target_actor.volatile = False
		current_actor = self.actor(state)

		actor_regularizer = self.criterion(current_actor, target_actor)
		actor_original_loss = -self.critic(state, self.actor(state)).mean()			

		# Compute actor total loss
		actor_loss = actor_original_loss + lambda_actor * actor_regularizer

		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		## For analysing actor regularisation
		loss_actor_original = actor_original_loss.data.cpu().numpy()
		loss_actor_regularizer = actor_regularizer.data.cpu().numpy()
		loss_actor = actor_loss.data.cpu().numpy()

		return loss_actor_regularizer, loss_actor_original, loss_actor

		

	def train(self, replay_buffer, iterations, batch_size=64, discount=0.99, tau=0.001, lambda_critic=0.1, lambda_actor=0.1):

		for it in range(iterations):

			loss_critic_regularizer, loss_critic_mse, loss_critic = self.critic_retrain(replay_buffer, iterations, batch_size, discount, tau, lambda_critic)
			loss_actor_regularizer, loss_actor_original, loss_actor = self.actor_update(replay_buffer, iterations, batch_size, discount, tau, lambda_actor)


			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

			for param, target_param,  in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

		return loss_critic_regularizer, loss_critic_mse, loss_critic, loss_actor_regularizer, loss_actor_original, loss_actor


	def save(self, filename, directory):
		torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
		torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))


	def load(self, filename, directory):
		self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
		self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))

