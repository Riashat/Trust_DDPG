import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math

import utils


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


	def forward(self, x, u):
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


	def train(self, replay_buffer, iterations, batch_size=64, discount=0.99, tau=0.001, lambda_critic=0.1, lambda_actor=0.1):

		for it in range(iterations):

			# Sample replay buffer 
			x, y, u, r, d = replay_buffer.sample(batch_size)
			state = var(torch.FloatTensor(x))
			action = var(torch.FloatTensor(u))
			next_state = var(torch.FloatTensor(y), volatile=True)
			done = var(torch.FloatTensor(1 - d))
			reward = var(torch.FloatTensor(r))

			"""
			Critic update
			"""
			# Q target = reward + discount * Q(next_state, pi(next_state))
			target_Q = self.critic_target(next_state, self.actor_target(next_state))
			#target_Q.volatile = False 
			target_Q = reward + (done * discount * target_Q)
			target_Q.volatile = False 

			# Get current Q estimate
			current_Q = self.critic(state, action)
			critic_mse = self.criterion(current_Q, target_Q)

			# target_Q_current_state = Variable(target_Q.data, requires_grad = True)
			target_Q_current_actor = self.critic_target(state, self.actor(state))
			target_Q_current_actor = Variable(target_Q_current_actor.data)
			target_Q_current_actor.volatile = False

			target_Q_target_actor = self.critic_target(state, self.actor_target(state))
			target_Q_target_actor = Variable(target_Q_target_actor.data)
			target_Q_target_actor.volatile = False

			critic_regularizer = self.criterion(target_Q_target_actor, target_Q_current_actor)

			# Compute critic total loss
			critic_loss = critic_mse + lambda_critic * critic_regularizer

			# Optimize the critic
			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			self.critic_optimizer.step()

			##for analysing regularisation
			loss_critic_regularizer = critic_regularizer.data.cpu().numpy()
			loss_critic_mse = critic_mse.data.cpu().numpy()
			loss_critic = critic_loss.data.cpu().numpy()


			"""
			Actor update
			"""
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

			##for analysing actor regularisation
			loss_actor_original = actor_original_loss.data.cpu().numpy()
			loss_actor_regularizer = actor_regularizer.data.cpu().numpy()
			loss_actor = actor_loss.data.cpu().numpy()


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



	# def train_with_adaptive_lambda(self, replay_buffer, iterations, batch_size=64, discount=0.99, tau=0.001, lambda_critic=0.1, lambda_actor=0.1):

	# 	for it in range(iterations):

	# 		# Sample replay buffer 
	# 		x, y, u, r, d = replay_buffer.sample(batch_size)
	# 		state = var(torch.FloatTensor(x))
	# 		action = var(torch.FloatTensor(u))
	# 		next_state = var(torch.FloatTensor(y), volatile=True)
	# 		done = var(torch.FloatTensor(1 - d))
	# 		reward = var(torch.FloatTensor(r))

	# 		"""
	# 		Critic update
	# 		"""
	# 		# Q target = reward + discount * Q(next_state, pi(next_state))
	# 		target_Q = self.critic_target(next_state, self.actor_target(next_state))
	# 		#target_Q.volatile = False 
	# 		target_Q = reward + (done * discount * target_Q)
	# 		target_Q.volatile = False 

	# 		# Get current Q estimate
	# 		current_Q = self.critic(state, action)
	# 		critic_mse = self.criterion(current_Q, target_Q)

	# 		# target_Q_current_state = Variable(target_Q.data, requires_grad = True)
	# 		target_Q_current_state = self.critic_target(state, self.actor(state))
	# 		target_Q_current_state = Variable(target_Q_current_state.data)
	# 		target_Q_current_state.volatile = False

	# 		critic_regularizer = self.criterion(current_Q, target_Q_current_state)

	# 		# Compute critic total loss
	# 		#First step: computing first term gradient magnitude
	# 		self.critic_optimizer.zero_grad()
	# 		critic_mse.backward(retain_graph=True)
	# 		#import pdb; pdb.set_trace()
	# 		grad_mag_sq = 0.0
	# 		for g in self.critic.parameters():
	# 			grad_mag_sq += torch.sum(g.grad.data**2)
	# 		grad_mag = torch.sqrt(torch.Tensor([grad_mag_sq]))[0]
	# 		#grad_mag_sq = np.sum([torch.sum(g.grad.data**2).data[0] for g in self.critic.parameters()])
	# 		#Second step: Computing "Lagrange multiplier" for critic regularizer...

	# 		critic_loss = critic_mse + math.sqrt(2 * lambda_critic)/grad_mag * critic_regularizer			

	# 		# Optimize the critic
	# 		self.critic_optimizer.zero_grad()
	# 		critic_loss.backward()
	# 		self.critic_optimizer.step()

	# 		##for analysing regularisation
	# 		loss_critic_regularizer = critic_regularizer.data.cpu().numpy()
	# 		loss_critic_mse = critic_mse.data.cpu().numpy()
	# 		loss_critic = critic_loss.data.cpu().numpy()

	# 		"""
	# 		Actor update
	# 		"""
	# 		target_actor = self.actor_target(state)
	# 		target_actor = Variable(target_actor.data)
	# 		target_actor.volatile = False
	# 		current_actor = self.actor(state)

	# 		actor_regularizer = self.criterion(current_actor, target_actor)
	# 		actor_original_loss = -self.critic(state, self.actor(state)).mean()			

	# 		# Compute actor total loss
	# 		#First step: computing first term gradient magnitude
	# 		self.actor_optimizer.zero_grad()
	# 		actor_original_loss.backward(retain_graph=True)
	# 		for g in self.actor.parameters():
	#  			grad_mag_sq += torch.sum(g.grad.data**2)
	# 		grad_mag = torch.sqrt(torch.Tensor([grad_mag_sq]))[0]
	# 		#grad_mag_sq = np.sum([torch.sum(g.grad.data**2) for g in self.actor.parameters()])
	# 		#Second step: Computing "Lagrange multiplier" for actor regularizer...
	# 		actor_loss = actor_original_loss + math.sqrt(2 * lambda_actor)/grad_mag * actor_regularizer

	# 		self.actor_optimizer.zero_grad()
	# 		actor_loss.backward()
	# 		self.actor_optimizer.step()

	# 		##for analysing actor regularisation
	# 		loss_actor_original = actor_original_loss.data.cpu().numpy()
	# 		loss_actor_regularizer = actor_regularizer.data.cpu().numpy()
	# 		loss_actor = actor_loss.data.cpu().numpy()


	# 		# Update the frozen target models
	# 		for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
	# 			target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

	# 		for param, target_param,  in zip(self.actor.parameters(), self.actor_target.parameters()):
	# 			target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

	# 	return loss_critic_regularizer, loss_critic_mse, loss_critic, loss_actor_regularizer, loss_actor_original, loss_actor


