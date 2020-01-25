import numpy as np
import gym
import torch
import torch.nn as nn
import argparse
import pickle

from functools import partial

state_vector = pickle.load(open("./data/state_vector_with_dyn_rand.p", "rb"))
state_labels = pickle.load(open("./data/state_labels_with_dyn_rand.p", "rb"))

s = np.arange(state_vector.shape[0])
np.random.shuffle(s)

state_vector = torch.from_numpy(state_vector[s])
state_labels = torch.from_numpy(state_labels[s])

state_vector_train = state_vector[0:500000]
state_labels_train = state_labels[0:500000][:,0] 

state_vector_test = state_vector[500000:750000]
state_labels_test = state_labels[500000:750000][:,0]

print(state_vector_train)
print(state_labels_train)

input_nodes, hidden_nodes, output_nodes, batch_size = 512, 512, 1, 100

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
        x = torch.relu(x)
        x = self.fc3(x)
        return x

model = Net()

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(10):
    permutation = torch.randperm(state_vector_train.shape[0])
    for data in range(0, state_vector_train.shape[0], batch_size):
        optimizer.zero_grad()
        # Forward Propagation
        indices = permutation[data:data+batch_size]
        batch_x, batch_y = state_vector_train[indices], state_labels_train[indices]
        batch_y = batch_y.unsqueeze(1)
        print(batch_x.shape, batch_y.shape)
        output = model(batch_x.float())    # Compute and print loss
        loss = criterion(output, batch_y.float())
        print('epoch: ', epoch, 'data: ', data,' loss: ', loss.item())    # Zero the gradients
        
        loss.backward()
        optimizer.step()

    pickle.dump(model, open("./trained_models_sl/Adam_10epoch_1e-4lr_500000:250000.p", "wb"))

model = pickle.load(open("./trained_models_sl/Adam_10epoch_1e-4lr_500000:250000.p", "rb"))
model.eval()


with torch.no_grad():
    predict = model(state_vector_test.float())
    loss = criterion(predict, state_labels_test.unsqueeze(1).float())
    print(loss)
for data in range(0, 1000):
    print("Predicted: ", predict[data], "Actual: ", state_labels_test[data], "Loss: ", loss.item())
