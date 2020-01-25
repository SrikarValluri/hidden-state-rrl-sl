import numpy as np
import gym
import torch
import torch.nn as nn
import argparse
import pickle

from functools import partial

state_vector = pickle.load(open("./data/state_vector_nonrandomized.p", "rb"))
state_labels = pickle.load(open("./data/state_labels_nonrandomized.p", "rb"))

state_vector_train = state_vector[0:500000]
state_labels_train = state_labels[0:500000][:,0] 

print(state_vector_train[0])
print(state_labels_train[0])