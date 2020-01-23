import numpy as np
import gym
import torch
import argparse
import pickle

from functools import partial

from cassie.cassie import CassieEnv_v2

parser = argparse.ArgumentParser()
parser.add_argument("--id", type=int, help="ID for parallelization")
args = parser.parse_args()

env = partial(CassieEnv_v2, traj='walking', simrate=60, clock_based=True, state_est=True, dynamics_randomization=True, no_delta=True, ik_traj=None)()
policy = torch.load('./trained_models_rrl/ccadea_randomized.pt')
policy.init_hidden_state()

state_vector = []
state_labels = []
#1000, 200

for i in range(1000):
    with torch.no_grad():
        visualize = False
        state = env.reset()
        done = False
        timesteps = 0
        eval_reward = 0
        if hasattr(policy, 'init_hidden_state'):
            policy.init_hidden_state()
        while not done and timesteps < 300:
            if hasattr(env, 'simrate'):            
                state = policy.normalize_state(state, update=False)
                action = policy.forward(torch.Tensor(state)).detach().numpy()
                state, reward, done, _ = env.step(action)
                hidden_state = np.ndarray.flatten(np.concatenate([np.array(policy.hidden[0]), np.array(policy.hidden[0])]))
                hidden_cells = np.ndarray.flatten(np.concatenate([np.array(policy.cells[0]), np.array(policy.cells[0])]))
                hidden_vector = np.concatenate([hidden_state, hidden_cells])
                if visualize:
                    env.render()
                eval_reward += reward
                timesteps += 1
                if timesteps > 249 and timesteps <= 299:
                    state_vector.append(hidden_vector.tolist())
                    state_labels.append([env.sim.get_body_ipos()[3], env.sim.get_ground_friction()[0], env.sim.get_ground_friction()[1]])

# print(np.array(state_vector).shape)
# print(np.array(state_labels).shape)
# print(np.array(state_labels))

state_vector = np.array(state_vector)
state_labels = np.ndarray.flatten(np.array(state_labels))
v = "state_vector" + str(args.id) + ".p"
l = "state_labels" + str(args.id) + ".p"

pickle.dump(state_vector, open("./data/" + v, "wb"))
pickle.dump(state_labels, open("./data/" + l, "wb"))
