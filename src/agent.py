import statistics

import numpy as np
from collections import deque
class Agent:

	def __init__(self, queue_length=100):
		self.reward_history = deque()
		self.queue_length = queue_length
		self.iteration = 0
	def add_reward(self, reward):
		self.reward_history.append(reward)
		if len(self.reward_history) > self.queue_length:
			self.reward_history.popleft()

	def avg_reward(self):
		if len(self.reward_history) > 0 and sum(self.reward_history) > 0:
			return statistics.mean(self.reward_history)
		return 0

	def act(self, env):
		raise NotImplementedError()




