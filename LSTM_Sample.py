import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import os, sys
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.a = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.b = nn.Linear(hidden_size, output_size)

    def forward(self, input, his=None):
        output, (hidden, cell) = self.a(input, his)
        output = torch.relu(output)
        output = torch.relu(self.b(output))

        return output, (hidden, cell)

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.orthogonal_(m.weight)
        m.bias.data.fill_(0.01)
    if type(m) == nn.LSTM:
        for name, param in m.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

'''
    불러온 모델에 기반해서 현재 모델을 초기화 하는 함수
    size 가 같다면 parameters 를 덮어 씌우지만 그렇지 않다면, 
    불러온 모델이 가진 parameters 의 mean, std 를 이용함 
    * 주의할 점은 초기화할 모델과 불러올 모델의 구조가 같아야 한다는 것 
    (i.e. 모델에 입력될 node size 만 다를 수 있음, layer 구조는 같아야 함)
'''
def init_weights_from_loaded(m, loaded_m):
    for name, param in m.named_parameters():
        if loaded_m.get(name) is not None:
            if param.shape == loaded_m[name].shape :
                param.data = loaded_m[name].data
            else :
                mean = torch.mean(loaded_m[name].data).item()
                std = torch.std(loaded_m[name].data).item()
                nn.init.normal_(param, mean=mean, std=std)

def save_model(m, path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save(m.state_dict(), path)


if __name__  == "__main__" :
    x = np.array([np.sin(x / 10) for x in range(100)], dtype='float');
    x += 1
    net = Net(input_size=1, hidden_size=50, output_size=1)
    net.apply(init_weights)
    optimizer = optim.SGD(params=net.parameters(), lr=5e-1)
    loss_function = nn.SmoothL1Loss()
    seq_length = 10
    curr_index = 0
    epochs = 10000
    history = []

    sliced_x = np.array([x[i: i + seq_length].reshape(seq_length, 1) for i in range(len(x) - seq_length)])
    sliced_y = np.array([x[i + 1: i + seq_length + 1].reshape(seq_length, 1) for i in range(len(x) - seq_length)])
    sliced_x = torch.from_numpy(sliced_x).float()
    sliced_y = torch.from_numpy(sliced_y).float()
    his = None

    for j in range(epochs):
        optimizer.zero_grad()
        output, _ = net(sliced_x, his)
        loss = loss_function(sliced_y, output)
        if j % 100 == 0:
            print(f'epoch {j} LOSS :  {loss.item()}')
        loss.backward()
        optimizer.step()
        history = []
    # 입력 데이터의 사이즈 (seq, batch, input)
    # 출력 데이터의 사이즈 (seq, batch, hidden)
    result = np.empty((len(x))); output = output.detach().numpy().reshape(-1, seq_length)
    result[0] = x[0]
    for i in range(0, len(x) - seq_length):
        result[i + 1:i + seq_length + 1] = output[i]
    plt.plot(x)
    plt.plot(result)
    plt.show()
    sys.exit(0)