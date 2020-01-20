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

env = partial(CassieEnv_v2, traj='walking', simrate=60, clock_based=True, state_est=True, dynamics_randomization=False, no_delta=False, ik_traj=None)()
policy = torch.load('./trained_models_rrl/actor_with_no_dyn_rand.pt')


state_vector = []
state_labels = []
#1000, 200
for i in range(1000):
    with torch.no_grad():
        state = env.reset()
        for j in range(200):
            state = policy.normalize_state(torch.Tensor(state))

            action = policy.forward(state, True)
            state, reward, done, info = env.step(np.ndarray.flatten(np.array(action)))
            hidden_state = np.ndarray.flatten(np.concatenate([np.array(policy.hidden[0]), np.array(policy.hidden[0])]))
            hidden_cells = np.ndarray.flatten(np.concatenate([np.array(policy.cells[0]), np.array(policy.cells[0])]))
            hidden_vector = np.concatenate([hidden_state, hidden_cells])
            env.render()
            if j > 149 and j <= 199:
                state_vector.append(hidden_vector.tolist())
                state_labels.append([env.sim.get_ground_friction()[0]])
print(np.array(state_vector).shape)
print(np.array(state_labels).shape)
# state_vector = np.array(state_vector)
# state_labels = np.ndarray.flatten(np.array(state_labels))
# v = "state_vector" + str(args.id) + ".p"
# l = "state_labels" + str(args.id) + ".p"

# pickle.dump(state_vector, open("./data/" + v, "wb"))
# pickle.dump(state_labels, open("./data/" + l, "wb"))
