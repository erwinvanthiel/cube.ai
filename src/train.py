from cube_environment import CubeEnvironment
from ppo_agent import PpoAgent


env = CubeEnvironment(2, 1)
agent = PpoAgent(env.get_state().shape[0], 12)
env.add_agent(agent)
EPISODES = agent.memory_size * 100000

for i in range(EPISODES):
    if i % agent.memory_size == 0:
        env.random_scramble()
    if i % (agent.memory_size * 1000) == 0:
        print(agent.avg_reward())
    env.next()
