'''
Demo: Linear Regression with Gradient Descent
      Regression: y = 2x + 2
'''
import csv

class LinearRegression:

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.k = 0
        self.b = 0
        self.J = self.loss(self.k, self.b)
        self.J_history = []

    def loss(self, k, b):
        return sum([(k * self.x[i] + b - self.y[i]) ** 2 for i in range(len(x))]) / 2 

    def gradient_descent(self, alpha, iterations):
        for i in range(iterations):
            self.k = self.k - alpha * sum([(self.k * x[i] + self.b - y[i]) * x[i] for i in range(len(x))])
            self.b = self.b - alpha * sum([(self.k * x[i] + self.b - y[i]) for i in range(len(x))])
            self.J_history.append(self.loss(self.k, self.b))
        return self.k, self.b

    def predict(self, x):
        return self.k * x + self.b

if __name__ == '__main__':

    x = [1, 2, 4, 4, 5, 6, 9, 9, 9, 10]
    y = [4, 5, 8, 12, 15, 16, 18, 21, 22, 24]

    model = LinearRegression(x, y)
    model.gradient_descent(0.0001, 1000)
    print('Regression Line: y = %f * x + %f' % (model.k, model.b))
