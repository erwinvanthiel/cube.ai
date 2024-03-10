import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.optim import Adam


class Actor(nn.Module):
    def __init__(self, alpha, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 512)
        self.ln1 = nn.LayerNorm(512)
        self.fc2 = nn.Linear(512, 256)
        self.ln2 = nn.LayerNorm(256)
        self.fc3 = nn.Linear(256, action_dim)

        self.optimizer = Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.fc3(x)
        return F.softmax(x, dim=-1)

    def save(self, episode):
        torch.save(self.state_dict(), '../model_params/actor-episode-{0}.pth'.format(episode))

    def load(self, path):
        self.load_state_dict(torch.load(path))


class Critic(nn.Module):
    def __init__(self, alpha, state_dim):
        super(Critic, self).__init__()
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 512)
        self.ln1 = nn.LayerNorm(512)
        self.fc2 = nn.Linear(512, 256)
        self.ln2 = nn.LayerNorm(256)
        self.fc3 = nn.Linear(256, 1)

        self.optimizer = Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.fc3(x)
        return x

    def save(self, episode):
        torch.save(self.state_dict(), '../model_params/critic-episode-{0}.pth'.format(episode))

    def load(self, path):
        self.load_state_dict(torch.load(path))