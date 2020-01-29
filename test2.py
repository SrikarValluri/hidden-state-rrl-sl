import numpy as np
import gym
import torch
import torch.nn as nn
import argparse
import pickle

from functools import partial

state_vector = pickle.load(open("./data/state_vector_ccadea_nonrandomized.p", "rb"))
state_labels = pickle.load(open("./data/state_labels_ccadea_nonrandomized.p", "rb"))

state_vector_train = state_vector[0:40000]
state_labels_train = state_labels[0:40000][:, 3]

state_vector_test = state_vector[40000:50000]
state_labels_test = state_labels[40000:50000][:, 3]

s = np.arange(state_vector_train.shape[0])
np.random.shuffle(s)

state_vector_train = torch.from_numpy(state_vector_train)
state_labels_train = torch.from_numpy(state_labels_train)

s = np.arange(state_vector_test.shape[0])
np.random.shuffle(s)

state_vector_test = torch.from_numpy(state_vector_test[s])
state_labels_test = torch.from_numpy(state_labels_test[s])

print(state_vector_train.shape)
print(state_labels_train.shape)

input_nodes, hidden_nodes, output_nodes, batch_size = 512, 512, 1, 100

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_nodes, hidden_nodes)
        self.fc2 = nn.Linear(hidden_nodes, hidden_nodes)
        self.fc3 = nn.Linear(hidden_nodes, output_nodes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.tanh(x)
        x = self.fc3(x)
        return x

criterion = nn.MSELoss()

model = pickle.load(open("./trained_models_sl/COM_Adam_1000epoch_5e-5lr_40000+10000_nonrandomized.p", "rb"))
model.eval()

with torch.no_grad():
    predict = model(state_vector_test.float())
    print(predict.size)
    loss = criterion(predict, state_labels_test.unsqueeze(1).float())
    print(loss)

percent = 0
for data in range(0, 10000):
    percent += torch.abs((predict[data] - state_labels_test[data])/state_labels_test[data] * 100)
    print("Predicted: ", predict[data], "Actual: ", state_labels_test[data], "Percentage: ", (predict[data] - state_labels_test[data])/state_labels_test[data] * 100, "%")
print("L1Loss: ", loss.item())
print("AvgPc: ", percent/10000)
print("nonrandomized.p")