# This represents the environment the agents are going to act in.
# This class is responsible for tracking state, having agents act and giving them rewards
class Environment:

    def __init__(self):
        self.agents = []
        self.iteration = 0
        self.state = None

    def add_agent(self, agent):
        self.agents.append(agent)
        agent.state = self.state

    def next(self):
        for agent in self.agents:
            agent.act(self)
        self.iteration += 1

    def perform_action(self, action):
        raise NotImplementedError()




