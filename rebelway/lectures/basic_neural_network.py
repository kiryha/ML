import numpy as np


def loss_function(w, b, train_data):

    result = 0
    count = len(train_data)

    for i in range(count):
        x = train_data[i][0]
        y = x*w + b
        d = y - train_data[i][1]

        result += d*d

    mean = result/count

    return mean


def train(train_data):
    
    np.random.seed(0)
    w = np.random.uniform(0, 10, 1)  # Weight
    b = np.random.uniform(0, 5, 1)  # Bias
    epsilon = 0.001
    epochs = 200
    learning_rate = 0.01

    print('LOSS: ', loss_function(w, b, train_data))

    # Forward pass
    for i in range(epochs):
        c = loss_function(w, b, train_data)
        cost_distance = (loss_function(w+epsilon, b, train_data) - c) / epsilon  # Simplified Derivative formula
        bias_distance = (loss_function(w, b+epsilon, train_data) - c) / epsilon

        w -= cost_distance * learning_rate
        b -= bias_distance * learning_rate

        if i % 10 == 0:
            print(loss_function(w, b, train_data))

    print('-'*20)
    print('w: ', w)  # w = 1.9 >> train_data[1 column] = train_data[0 column] * 2 <-- (2 ~= 1.9)
    print('b: ', b)


if __name__ == "__main__":

    train_data = [[0, 0],
                  [1, 2],
                  [2, 4],
                  [4, 8],
                  [8, 16]]

    train(train_data)
