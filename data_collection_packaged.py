import numpy as np
import torch
import argparse
import pickle

state_vector = []
state_labels = []

policy_hash = 'ccadea_randomized.pt'

for i in range(25):
    v = "state_vector_" + policy_hash[:-3] + "-" + str(i + 1) + ".p"
    l = "state_labels_" + policy_hash[:-3] + "-" + str(i + 1) + ".p"
    v = (pickle.load(open("./data/" + v, "rb" ))).tolist()
    l = (pickle.load(open("./data/" + l, "rb" ))).tolist()
    state_vector.append(v)
    state_labels.append(l)

print(np.ndarray.flatten(np.array(v)).shape)
state_vector = np.reshape(np.ndarray.flatten(np.array(state_vector)), (-1, 512))
state_labels = np.reshape(np.ndarray.flatten(np.array(state_labels)), (-1, 3))

print(state_vector.shape)
print(state_labels.shape)

pickle.dump(state_vector, open("./data/" + "state_vector_" + policy_hash[:-3] + ".p", "wb"))
pickle.dump(state_labels, open("./data/" + "state_labels_" + policy_hash[:-3] + ".p", "wb"))
