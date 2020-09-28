import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Define model
class TheModelClass(nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        super(Net, self).__init__()
        self.a = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.b = nn.Linear(hidden_size, output_size)
        self.optimizer = optim.RMSprop(lr=learning_rate, params=self.parameters())

    def forward(self, input, his=None):
        output, (hidden, cell) = self.a(input, his)
        # output [ :, -1, : ] 와 hidden 값은 같다.
        output = output[:, -1, :]
        # output = torch.relu(output)
        output = self.b(output)

        return output, (hidden, cell)


# Initialize model
model = TheModelClass()
model2 = Net(16, 50, 8, 1e-4)
model3 = Net(2, 100, 1, 1e-4)
loaded_model = torch.load('model.pt')
# Initialize optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
for param_tensor in model2.state_dict():
    print(param_tensor, "\t", model2.state_dict()[param_tensor].size())
    print(torch.mean(model2.state_dict()[param_tensor]), torch.std(model2.state_dict()[param_tensor]))
for param_tensor in model3.state_dict():
    print(param_tensor, "\t", model3.state_dict()[param_tensor].size())
for param_tensor in loaded_model.state_dict():
    print(param_tensor)

torch.save(model2.state_dict(), './model.pt')

# Print optimizer's state_dict
print("The state_dict of the optimizers :")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])
