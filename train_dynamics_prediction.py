import numpy as np
import gym
import torch
import torch.nn as nn
import argparse
import pickle

from functools import partial

state_vector = pickle.load(open("./data/state_vector_ccadea_randomized.p", "rb"))
state_labels = pickle.load(open("./data/state_labels_ccadea_randomized.p", "rb"))

s = np.arange(state_vector.shape[0])
np.random.shuffle(s)

state_vector = torch.from_numpy(state_vector[s])
state_labels = torch.from_numpy(state_labels[s])

state_vector_train = state_vector[0:40000]
state_labels_train = state_labels[0:40000] 

state_vector_test = state_vector[40000:50000]
state_labels_test = state_labels[40000:50000]

print(state_vector_train.shape)
print(state_labels_train.shape)

input_nodes, hidden_nodes, output_nodes, batch_size = 512, 512, 139, 100

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

model = Net()

criterion = nn.L1Loss()

optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

for epoch in range(200):
    permutation = torch.randperm(state_vector_train.shape[0])
    for data in range(0, state_vector_train.shape[0], batch_size):
        optimizer.zero_grad()
        # Forward Propagation
        indices = permutation[data:data+batch_size]
        batch_x, batch_y = state_vector_train[indices], state_labels_train[indices]
        batch_y = batch_y.unsqueeze(1)
        # print(batch_x.shape, batch_y.shape)
        output = model(batch_x.float())    # Compute and print loss
        print(output.shape, batch_y.float().shape)
        loss = criterion(output, batch_y.float())
        print('epoch: ', epoch, 'data: ', data,' loss: ', loss.item())    # Zero the gradients
        
        loss.backward()
        optimizer.step()

    pickle.dump(model, open("./trained_models_sl/Allparams_Adam_10epoch_1e-4lr_40000:50000_randomized.p", "wb"))

model = pickle.load(open("./trained_models_sl/Allparams_Adam_10epoch_1e-4lr_40000:50000_randomized.p", "rb"))
model.eval()


with torch.no_grad():
    predict = model(state_vector_test.float())
    print(predict.size)
    loss = criterion(predict, state_labels_test.unsqueeze(1).float())
    print(loss)

difference = 0
for data in range(0, 1000):
    difference += np.abs(predict[data][0] - state_labels_test[data])
    print("Predicted: ", predict[data], "Actual: ", state_labels_test[data])
print("L1Loss: ", loss.item())