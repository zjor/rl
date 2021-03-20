import csv
import random
import numpy as np
from dqn import Model


def load_q_table(filename):
    data = []
    for row in csv.reader(open(filename, "r")):
        data.append(list(map(float, row)))

    x = []
    y = []
    for i, row in enumerate(data):
        _x = np.zeros(len(data))
        _x[i] = 1
        x.append(list(_x))

        y.append(list(np.array(row)))
    return x, y


def up_sample(x, y, n):
    u_x = []
    u_y = []
    for j in range(n):
        for i in range(len(x)):
            u_x.append(x[i])
            u_y.append(y[i])

    random.shuffle(u_x)
    random.shuffle(u_y)
    return u_x, u_y


def predict(model, s_index):
    _x = np.zeros(12)
    _x[s_index] = 1
    with torch.no_grad():
        return model(torch.FloatTensor(_x))


def print_policy(model):
    actions = ['←', '→', '↑', '↓']
    w, h = 4, 3
    for y in range(h):
        for x in range(w):
            s = y * w + x
            y_pred = predict(model, s)
            a = torch.argmax(y_pred, 0).item()
            print(actions[a], end='')
        print()


if __name__ == "__main__":
    import sys
    import torch
    from torch import nn
    from sklearn.model_selection import train_test_split
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    x, y = load_q_table("./q_table.csv")
    print(x)
    print(y)
    sys.exit(1)
    # x, y = up_sample(x, y, 10)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=33)
    x_train = torch.FloatTensor(x_train)
    x_test = torch.FloatTensor(x_test)
    y_train = torch.FloatTensor(y_train)
    y_test = torch.FloatTensor(y_test)

    in_features = len(x_train[0])
    out_features = len(y_train[0])
    print(in_features, out_features)
    model = Model(in_features=in_features, hidden=[], out_features=out_features)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    epochs = 500
    losses = []

    for i in range(epochs):
        y_pred = model.forward(x_train)
        loss = criterion(y_pred, y_train)
        losses.append(loss)

        if i % 100 == 1:
            print(f'epoch: {i:2}  loss: {loss.item():10.8f}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print_policy(model)
    print(predict(model, 0))
