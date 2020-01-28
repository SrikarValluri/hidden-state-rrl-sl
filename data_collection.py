import numpy as np
import gym
import torch
import argparse
import pickle

from functools import partial

from cassie.cassie import CassieEnv_v2

policy_hash = 'ccadea_nonrandomized.pt'

parser = argparse.ArgumentParser()
parser.add_argument("--id", type=int, help="ID for parallelization")
args = parser.parse_args()

env = partial(CassieEnv_v2, traj='walking', simrate=60, clock_based=True, state_est=True, dynamics_randomization=True, no_delta=True, ik_traj=None)()
policy = torch.load('./trained_models_rrl/' + policy_hash)

state_vector = []
state_labels = []
#1000, 200

for i in range(2000):
    with torch.no_grad():
        visualize = False
        state = env.reset()
        done = False
        timesteps = 0
        eval_reward = 0
        if hasattr(policy, 'init_hidden_state'):
            policy.init_hidden_state()
        while timesteps < 150 and not done:
            if hasattr(env, 'simrate'):            
                state = policy.normalize_state(state, update=False)
                action = policy.forward(torch.Tensor(state)).detach().numpy()
                state, reward, done, _ = env.step(action)
                if visualize:
                    env.render()
                eval_reward += reward
                timesteps += 1
        hidden_state = np.ndarray.flatten(np.concatenate([np.array(policy.hidden[0]), np.array(policy.hidden[0])]))
        hidden_cells = np.ndarray.flatten(np.concatenate([np.array(policy.cells[0]), np.array(policy.cells[0])]))
        hidden_vector = np.concatenate([hidden_state, hidden_cells])
        state_vector.append(hidden_vector.tolist())
        # node_values = policy._get_hidden_layers(torch.Tensor(state))
        # state_vector.append(node_values.tolist())
        state_labels.append(np.concatenate([env.sim.get_body_ipos(), env.sim.get_ground_friction(), env.sim.get_dof_damping(), env.sim.get_body_mass()]).tolist())
        # print(len(np.concatenate([env.sim.get_body_ipos(), env.sim.get_dof_damping(), env.sim.get_body_mass(), env.sim.get_ground_friction()]).tolist()))
        print(env.sim.get_ground_friction().shape)
state_vector = np.array(state_vector)
state_labels = np.array(state_labels)
v = "state_vector_" + policy_hash[:-3] + "-" + str(args.id) + ".p"
l = "state_labels_" + policy_hash[:-3] + "-" + str(args.id) + ".p"

pickle.dump(state_vector, open("./data/" + v, "wb"))
pickle.dump(state_labels, open("./data/" + l, "wb"))
