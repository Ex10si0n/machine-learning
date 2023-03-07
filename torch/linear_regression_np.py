import numpy as np
import matplotlib.pyplot as plt

def forward(x):
    return x * w

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)

def gradient(x, y):
    return 2 * x * (x * w - y)

def iterate(w, x_data, y_data):

    w_list = []
    mse_list = []
    for w in np.arange(0.0, 4.1, 0.1):
        print("w = ", w)
        l_sum = 0
        for x_val, y_val in zip(x_data, y_data):
            y_pred_val = forward(x_val)
            l = loss(x_val, y_val)
            l_sum += l
            print("\t", x_val, y_val, y_pred_val, l)
        print("MSE = ", l_sum / 3)
        w_list.append(w)
        mse_list.append(l_sum / 3)
    plt.plot(w_list, mse_list)
    plt.ylabel('Loss')
    plt.xlabel('w')
    plt.show()

def gradient_descend(w, x_data, y_data):
    print("Predict (before training)", 4, forward(4))

    for epoch in range(100):
        for x_val, y_val in zip(x_data, y_data):
            grad = gradient(x_val, y_val)
            w = w - 0.01 * grad
            # print("\tgrad: ", x_val, y_val, grad)
            l = loss(x_val, y_val)

        print("[ epoch :", epoch, "]   w =", w, "  loss =", l)

    print("Predict (after training)", "4 hours", forward(4))

if __name__ == "__main__":
    x_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    y_data = [2.3, 2.6, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
    w = 1.0

    gradient_descend(w, x_data, y_data)
