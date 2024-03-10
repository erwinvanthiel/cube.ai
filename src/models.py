import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.optim import Adam


class Actor(nn.Module):
    def __init__(self, alpha, state_dim, action_dim, hidden_size=512):
        super(Actor, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Softmax(dim=-1)
        )

        self.optimizer = Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        return self.actor(state)

    def save(self, episode):
        torch.save(self.state_dict(), '../model_params/actor-episode-{0}.pth'.format(episode))

    def load(self, path):
        self.load_state_dict(torch.load(path))


class Critic(nn.Module):
    def __init__(self, alpha, state_dim, hidden_size=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

        self.optimizer = Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value

    def save(self, episode):
        torch.save(self.state_dict(), '../model_params/critic-episode-{0}.pth'.format(episode))

    def load(self, path):
        self.load_state_dict(torch.load(path))