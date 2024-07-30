import numpy as np
import pandas as pd

def loss_function(w1, w2, b, train_data):

    result = 0
    count = len(train_data)

    for i in range(count):
        x1 = train_data[i][0]
        x2 = train_data[i][1]
        y = sigmoid(x1 * w1 + x2 * w2 + b)
        d = y - train_data[i][2]
        result += d * d

    mean = result / count

    return mean


def sigmoid(x):

    return 1/(1 + np.exp(-x))


def train(train_data):

    # np.random.seed(0)

    epochs = 5000
    epsilon = 0.01
    learning_rate = 0.1
    loss_values = []

    w1 = np.random.uniform(0, 1, 1)  # Weight
    w2 = np.random.uniform(0, 1, 1)  # Weight
    b = np.random.uniform(0, 1, 1)  # Bias

    print('Init Loss: ', loss_function(w1, w2, b, train_data))

    for i in range(epochs):

        c = loss_function(w1, w2, b, train_data)
        loss_values.append(c)
        # print(f'w1: {w1} w2: {w2} loss: {c}')

        dw1 = (loss_function(w1 + epsilon, w2, b, train_data) - c) / epsilon
        dw2 = (loss_function(w1, w2 + epsilon, b, train_data) - c) / epsilon
        bd = (loss_function(w1, w2, b + epsilon, train_data) - c) / epsilon

        w1 -= learning_rate * dw1
        w2 -= learning_rate * dw2
        b -= learning_rate * bd

    print('---------------')
    print(f'w1: {w1} w2: {w2}  b: {b} loss: {loss_function(w1, w2, b, train_data)}')
    print('---------------')

    # Test predictions
    for i in range(2):
        for j in range(2):
            print(f'i {i} | j {sigmoid(i*w1 + j*w2 + b)} ')

    return loss_values


if __name__ == "__main__":

    # Train Data row = [feature 1, feature 2, label (prediction)]

    train_data = [[0, 0, 0],
                  [1, 0, 1],
                  [0, 1, 1],
                  [1, 1, 1]]

    loss_values = train(train_data)
    dataframe = pd.DataFrame(loss_values)
    dataframe.to_csv('two_features_lost_values.csv', index=False)
