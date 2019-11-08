import random

NUM_ITER = 100
BATCH_SIZE = 10

class BasicMDP:
	"""This is a placeholder MDP"""

	def __init__(self):
		self.states = range(10)
		self.reward = [0 for i in range(10)]
		self.reward[9] = 10
		self.actions = [-1, 1]
		self.Q = [[0.0 for i in range(10)] for j in range(2)]
		self.start = 0

	def _policy(self, state, action):
		"""Random policy"""
		i = random.random()
		return i/(state + action)

	def transition(self, state, action):
		p = random.random()
		if p > 0.01:
			return state + action
		else:
			return state - action


class Samples:
	"""Class holds state"""

	def __init__(self, MDP, theta):
		self.MDP = MDP
		self.record = []
		self.state = MDP.start
		self.theta = theta

	def step(self):

		action = self.MDP.policy(self.state)
		self.record.append((self.state, self.action))
		self.state = self.MDP.transition(self.state, action)

	def run(self, iters):

		for i in range(iters):
			step()
		return self.record


mdp = BasicMDP()

for j in range(NUM_ITER):
	# run MDP to collect samples
	runtime = Samples(mdp)
	samples = runtime.run(BATCH_SIZE)

	# what is the baseline function???

	# calculate advantages
	advantages = []
	for i in range(BATCH_SIZE):
		st = samples[i][0]
		act = samples[i][1]
		adv = mdp.policy(st, act)
		

