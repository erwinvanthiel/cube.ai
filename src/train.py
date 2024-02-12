from src.cube_environment import CubeEnvironment
from src.ppo_agent import PpoAgent


env = CubeEnvironment(3)
agent = PpoAgent(env.get_flat_state().shape[0], 18)
env.add_agent(agent)
EPISODES = agent.memory_size * 100000

for i in range(EPISODES):
    if i % agent.memory_size == 0:
        env.random_scramble(10)
    if i % (agent.memory_size * 10) == 0:
        print(agent.avg_reward())
    env.next()
