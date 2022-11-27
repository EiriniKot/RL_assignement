import json
import gym
import numpy as np
import torch
from agent import DqnAgent
from network import DqnNet
from buffer import ReplayMemory, Transition
from itertools import count
from matplotlib import pyplot as plt


class GamesRunner:
    def __init__(self, specs, h, w, batch=10, capacity=100, num_episodes=10):
        self.envs = {}
        for env in specs['train_envs']:
            self.envs[env] = gym.make(env)
        n_actions = self.envs[env].action_space.n

        self.h = h
        self.w = w
        self.batch =batch

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.capacity = capacity

        self.num_episodes = num_episodes
        self.r_buffer = ReplayMemory(self.capacity)
        self.network_obj = DqnNet(self.h, self.w, n_actions)

        self.agent = DqnAgent(self.network_obj, n_actions, device=self.device,
                              optimizer=specs['optimizer'], **specs['policy_specs'])

    def run(self):
        for env_n, env in self.envs.items():
            print(f'Environment {env_n}')
            for ep in range(self.num_episodes):
                # Initialize the environment and state
                state = env.reset()
                # BCHW
                state = np.transpose(state, (2, 0, 1))
                state = np.expand_dims(np.resize(state, (3, self.h, self.w)), 0)
                state = torch.from_numpy(state).type(torch.float32)

                for t in count():
                    # Select and perform an action
                    action = self.agent.policy(state)
                    next_state, reward, done, info = env.step(action.item())
                    # BCHW
                    next_state = np.transpose(next_state, (2, 0, 1))
                    next_state = np.expand_dims(np.resize(next_state, (3, self.h, self.w)), 0)
                    next_state = torch.from_numpy(next_state).type(torch.float32)
                    reward = torch.tensor([reward], device=self.device)
                    # Store the transition in memory
                    self.r_buffer.push(state, action, next_state, reward)
                    state = next_state
                    if len(self.r_buffer) >= self.batch:
                        transitions = self.r_buffer.sample(self.batch)
                        experience = Transition(*zip(*transitions))
                        # Perform one step of the optimization (on the policy network)
                        self.agent.train(experience)
                    self.agent.update_target(t)

                    if done:
                        print('----DONE-----')
                        break
            plt.plot(self.agent.loss_saver)
            plt.show()


if __name__ == '__main__':
    f = open('envs.json')
    json_config = json.load(f)
    runner = GamesRunner(json_config, h=100, w=100, capacity=100, num_episodes=2)
    runner.run()





