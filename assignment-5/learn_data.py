import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd
import numpy as np

from tensorboardX import SummaryWriter
writer = SummaryWriter()

batch_size = 100

class Network(nn.Module):
    def __init__(self, input_size):
        super(Network, self).__init__()
        self.input_layer = nn.Linear(input_size, 256)
        self.hidden_layer_1 = nn.Linear(256, 128)
        self.hidden_layer_2 = nn.Linear(128, 128)
        self.hidden_layer_3 = nn.Linear(128, 128)
        self.output_layer = nn.Linear(128, 6)

        self.dropout1 = nn.Dropout(0.6)
        self.dropout2 = nn.Dropout(0.6) 
        self.dropout3 = nn.Dropout(0.6) 
        self.dropout4 = nn.Dropout(0.6) 

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = self.dropout1(x)
        x = F.relu(self.hidden_layer_1(x))
        x = self.dropout2(x)
        x = F.relu(self.hidden_layer_2(x))
        x = self.dropout3(x)
        x = F.relu(self.hidden_layer_3(x))
        x = self.dropout4(x)
        x = F.softmax(self.output_layer(x))
        return x

labels = ['CLASSIFICATION_Accident', 'CLASSIFICATION_Misdemeanor', 'CLASSIFICATION_Other', 'CLASSIFICATION_Service', 'CLASSIFICATION_Theft', 'CLASSIFICATION_Violence']

training_set_df = pd.read_csv('training_set_one_hot.csv', engine='python')
training_set_labels_df = training_set_df[labels]
training_set_df = training_set_df.drop(columns=labels)
training = TensorDataset(torch.from_numpy(np.array(training_set_df)), torch.from_numpy(np.array(training_set_labels_df)))
training_loader = DataLoader(training, batch_size=batch_size, shuffle=True)

validation_set_df = pd.read_csv('validation_set_one_hot.csv', engine='python')
validation_set_labels_df = validation_set_df[labels]
validation_set_df = validation_set_df.drop(columns=labels)
validation = TensorDataset(torch.from_numpy(np.array(validation_set_df)), torch.from_numpy(np.array(validation_set_labels_df)))
validation_loader = DataLoader(validation, batch_size=batch_size, shuffle=True)

test_set_df = pd.read_csv('test_set_one_hot.csv', engine='python')
test_set_labels_df = test_set_df[labels]
test_set_df = test_set_df.drop(columns=labels)
test = TensorDataset(torch.from_numpy(np.array(test_set_df)), torch.from_numpy(np.array(test_set_labels_df)))
test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

print(len(training_set_df.columns))
model = Network(len(training_set_df.columns))

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

n_epochs = 100

valid_loss_min = np.Inf

for epoch in range(n_epochs):
    train_loss = 0.0
    valid_loss = 0.0

    model.train()

    train_count = 0
    valid_count = 0

    for data, target in training_loader:
        optimizer.zero_grad()
        output = model(data.float())
        loss = criterion(output.float(), target.float())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*data.size(0)
        train_count += 1
        
    model.eval()
    for data, target in validation_loader:
        output = model(data.float())
        loss = criterion(output.float(), target.float())
        valid_loss += loss.item()*data.size(0)
        valid_count += 1
    
    train_loss = train_loss / train_count
    valid_loss = valid_loss / valid_count

    writer.add_scalars('loss/train+valid', {'train': train_loss, 'valid': valid_loss}, epoch)
        
    print(f'Epoch: {epoch} \tTrain Loss: {train_loss} \tValid Loss: {valid_loss}')
    

model.eval()
test_loss=0
correct=0
for data,target in test_loader:
    output = model(data.float())
    for i in range(len(target)):
        if target[i][output[i].max(0)[1]] == 1:
            correct += 1

print('\nAccuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset)}%)\n')