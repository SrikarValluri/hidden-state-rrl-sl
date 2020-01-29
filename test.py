import numpy as np
import gym
import torch
import torch.nn as nn
import argparse
import pickle

from functools import partial

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.tanh(x)
        x = self.fc3(x)
        return x

model_randomized = pickle.load(open("./trained_models_sl/COM_Adam_1000epoch_5e-5lr_40000+10000_randomized.p", "rb"))
model_nonrandomized = pickle.load(open("./trained_models_sl/COM_Adam_1000epoch_5e-5lr_40000+10000_nonrandomized.p", "rb"))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(600, 600)
        self.fc2 = nn.Linear(600, 600)
        self.fc3 = nn.Linear(600, 1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.tanh(x)
        x = self.fc3(x)
        return x

model_forward = pickle.load(open("./trained_models_sl/COM_Adam_1000epoch_5e-5lr_40000+10000_forward.p", "rb"))

model_randomized.eval()
model_nonrandomized.eval()
model_forward.eval()

# Evaluation Data

state_vector_randomized = torch.from_numpy(pickle.load(open("./data/state_vector_ccadea_randomized.p", "rb")))[40000:50000]
state_labels_randomized = torch.from_numpy(pickle.load(open("./data/state_labels_ccadea_randomized.p", "rb")))[40000:50000][:, 3]

state_vector_nonrandomized = torch.from_numpy(pickle.load(open("./data/state_vector_ccadea_nonrandomized.p", "rb")))[40000:50000]
state_labels_nonrandomized = torch.from_numpy(pickle.load(open("./data/state_labels_ccadea_nonrandomized.p", "rb")))[40000:50000][:, 3]

state_vector_forward = torch.from_numpy(pickle.load(open("./data/state_vector_ccadea_forward.p", "rb")))[40000:50000]
state_labels_forward = torch.from_numpy(pickle.load(open("./data/state_labels_ccadea_forward.p", "rb")))[40000:50000][:, 3]


predict_randomized = model_randomized(state_vector_randomized.float())
predict_nonrandomized = model_nonrandomized(state_vector_nonrandomized.float())
predict_forward = model_forward(state_vector_forward.float())

err_null = torch.mean(state_labels_nonrandomized)
e_null = 0
e_randomized = 0
e_nonrandomized = 0
e_forward = 0
percent = 0
for data in range(0, 10000):
    err_randomized = predict_randomized[data]
    err_nonrandomized = predict_nonrandomized[data]
    err_forward = predict_forward[data]
    
    e_null += 1/10000 * (torch.abs(err_null - state_labels_nonrandomized[data]))
    e_randomized += 1/10000 * (torch.abs(err_randomized - state_labels_randomized[data]))
    e_nonrandomized += 1/10000 * (torch.abs(err_nonrandomized - state_labels_nonrandomized[data]))
    e_forward += 1/10000 * (torch.abs(err_forward - state_labels_forward[data]))
    print(data)
    # percent_per_iter = (predict_randomized[data] - state_labels_randomized[data])/state_labels_randomized[data] * 100
    # percent += percent_per_iter
    # print("Predicted: ", predict_randomized[data], "Actual: ", state_labels_randomized[data], "Percentage: ", percent_per_iter, "%")

print(percent/10000)
print(e_null)
print(e_randomized)
print(e_nonrandomized)
print(e_forward)

print("PercentRandomized: ", (e_null - e_randomized)/e_null * 100, "%") 
print("PercentNonRandomized: ", (e_null - e_nonrandomized)/e_null * 100, "%") 
print("PercentForward: ", (e_null - e_forward)/e_null * 100, "%") 

