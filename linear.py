import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader


array = [[i, i, i] for i in range(100)]
# print(array)

x = np.array(array, dtype=float)

w = np.array([[0.1], [0.2], [0.3]], dtype=float)

label = x.dot(w) + 1
# print(label)

y = label + np.random.normal(0, 0.01)


class MyData(Dataset):
    def __init__(self):
        super(MyData, self).__init__()
        self.data = torch.from_numpy(x).to(torch.float32)
        self.target = torch.from_numpy(label).to(torch.float32)

    def __getitem__(self, item):
        self.x = self.data[item]
        self.label = self.target[item]

        return self.x, self.label

    def __len__(self):
        return len(self.data)


class MyLinear(nn.Module):
    def __init__(self):
        super(MyLinear, self).__init__()
        w = torch.tensor([[0.12], [0.23], [0.34]], dtype=torch.float32, requires_grad=True)
        b = torch.tensor([2], dtype=torch.float32, requires_grad=True)

        self.w = torch.nn.Parameter(w)
        self.b = torch.nn.Parameter(b)

        self.register_parameter("weight", self.w)
        self.register_parameter("bias", self.b)

    def forward(self, in_x):
        return in_x.matmul(self.w) + self.b


dataset = MyData()
dataloader = DataLoader(dataset, shuffle=True, batch_size=4)

epochs = 1000
model = MyLinear()
# for p in model.parameters():
#     print(p)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()


for epoch in range(epochs):
    model.train()
    for data in dataloader:
        in_x, out_label = data

        optimizer.zero_grad()
        # print(model(in_x))
        # print("______________________________________")
        # print(out_label)
        loss = loss_fn(model(in_x), out_label).sum()
        loss.backward()
        optimizer.step()

    print("epoch{}  ".format(epoch + 1))
    for p in model.parameters():

        print(p)
    print(loss)

