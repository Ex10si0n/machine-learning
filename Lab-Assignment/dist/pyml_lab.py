class LinearRegression:

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.k = 0
        self.b = 0
        self.J = self.loss(self.k, self.b)
        self.J_history = []

    def loss(self, k, b):
        # Tips here:
      	# 	program the loss function with given formula
        # 	where x is self.x, y is self.y from class LinearRegression
        # 	which means you can get x_i by for loop all of the indices of 0 ... len(x)
      	loss = _____________________
        return loss

    def gradient_descent(self, alpha, iterations):
        for i in range(iterations):
          	# Tips here:
            # 	program the partial derivatives dk, db
            #		refer the formula from section [Calculating Partial derivatives]
            # 	where k is self.k, b is self.b from class LinearRegression
            # 	notice that the code here is already inside for loop
          	dk = _____________________
            db = _____________________
            self.k = self.k - alpha * dk
            self.b = self.b - alpha * db
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
