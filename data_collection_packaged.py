import numpy as np
import torch
import argparse
import pickle

state_vector = []
state_labels = []

for i in range(15):
    v = "state_vector" + str(i + 1) + ".p"
    l = "state_labels" + str(i + 1) + ".p"
    v = (pickle.load(open("./data/" + v, "rb" ))).tolist()
    l = (pickle.load(open("./data/" + l, "rb" ))).tolist()
    state_vector.append(v)
    state_labels.append(l)

print(np.ndarray.flatten(np.array(v)).shape)
state_vector = np.reshape(np.ndarray.flatten(np.array(state_vector)), (-1, 512))
state_labels = np.reshape(np.ndarray.flatten(np.array(state_labels)), (-1, 3))

print(state_vector.shape)
print(state_labels.shape)

pickle.dump(state_vector, open("./data/" + "state_vector_no_dyn_random.p", "wb"))
pickle.dump(state_labels, open("./data/" + "state_labels_no_dyn_random.p", "wb"))
