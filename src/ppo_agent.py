from torch.distributions import Categorical
from agent import Agent
from models import Actor, Critic
import torch
import numpy as np
import random
from plot import plot

class PpoAgent(Agent):

    def __init__(self, state_dim, num_actions, n_epochs=4, memory_size=16, batch_size=8, policy_clip=0.2, gamma=0.99, gae_lambda=0.95, alpha=0.000005, debug=False):
        super(PpoAgent, self).__init__()
        self.train = True
        self.action_probabilties = np.empty(memory_size)
        self.values = np.empty(memory_size)
        self.actions_taken = np.empty(memory_size)
        self.rewards = np.empty(memory_size)
        self.dones = np.empty(memory_size)
        self.actor = Actor(alpha, state_dim, num_actions)
        self.critic = Critic(alpha, state_dim)
        self.states = np.empty((memory_size, state_dim))
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.total_reward = 0
        self.debug = debug

    def act(self, env):
        # Check whether memory is full and perform update if so
        if self.iteration == self.memory_size:
            if self.debug:
                print("---------- episode ----------")
            self.learn()
            self.iteration = 0
            self.add_reward(self.total_reward)
            self.total_reward = 0

        if self.debug:
            print(env.get_state())

        self.states[self.iteration] = env.get_state()
        # the policy output, aka a probability distribution
        probs = self.actor(torch.tensor(env.get_state()).float().cuda())
        pi = Categorical(probs) 
        
        if self.debug:
            print(probs)

        # the state value approximation, i.e. the Q-value approximation.
        self.values[self.iteration] = self.critic(torch.tensor(env.get_state()).float().cuda())

        # the sampled action
        a = pi.sample()

        if self.debug:
            print(a.item())

        # a = torch.tensor([random.Random().randint(0, 11)]).cuda() # FOR DEBUGGING
        self.actions_taken[self.iteration] = a

        # the probability of the sampled action
        self.action_probabilties[self.iteration] = pi.log_prob(a)

        # perform the action
        reward, self.dones[self.iteration] = env.perform_action(a.item())
        self.rewards[self.iteration] = reward
        self.total_reward += reward

        self.iteration += 1


    # Implementation based on
    # https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/PolicyGradient/PPO/torch/ppo_torch.py
    def learn(self):
        for _ in range(self.n_epochs):

            # Calculate advantages with GAE
            A = np.zeros(len(self.rewards), dtype=np.float32)
            for t in range(len(self.rewards) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(self.rewards) - 1):
                    a_t += discount * (self.rewards[k] + self.gamma * self.values[k + 1] * \
                                       (1 - int(self.dones[k])) - self.values[k])
                    discount *= self.gamma * self.gae_lambda
                A[t] = a_t
            A = torch.tensor(A).to(self.actor.device)

            values = torch.tensor(self.values).to(self.actor.device)

            for batch in self.create_random_batches():
                states = torch.tensor(self.states[batch], dtype=torch.float).to(self.actor.device)
                pi_old = torch.tensor(self.action_probabilties[batch]).to(self.actor.device)
                actions = torch.tensor(self.actions_taken[batch]).to(self.actor.device)

                dist = Categorical(self.actor(states))
                critic_value = self.critic(states)
                critic_value = torch.squeeze(critic_value)

                # Compare old and new probs and use advantage to calculate actor loss
                pi_new = dist.log_prob(actions)
                prob_ratio = pi_new.exp() / pi_old.exp()
                weighted_probs = A[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.policy_clip,
                                                 1 + self.policy_clip) * A[batch]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                # Calculate the return of each state and use MSELoss to update the critic
                returns = A[batch] + values[batch]
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5 * critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

    def create_random_batches(self):
        indices = np.arange(self.memory_size, dtype=np.int64)
        random.shuffle(indices)
        groups = [indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)]
        return groups
