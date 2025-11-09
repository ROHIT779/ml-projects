import numpy as np
import copy
class LogisticRegression:
  
  def sigmoid(self, x):
    return 1 / (1 + np.exp(-x))

  def compute_cost(self, X, y, w, b, lambda_ = 0):
    m, n = len(X), len(X[0])
    cost = 0.
    for i in range(m):
      z_wb_i = np.dot(X[i], w) + b
      f_wb_i = self.sigmoid(z_wb_i)
      cost += (f_wb_i - y[i]) ** 2
    cost /= (2 * m)
    return cost

  def compute_gradient(self, X, y, w, b, lambda_ = 0):
    m, n = len(X), len(X[0])
    dj_dw = np.zeros((n,))
    dj_db = 0.
    for i in range(m):
      z_wb_i = np.dot(w, X[i]) + b
      f_wb_i = self.sigmoid(z_wb_i)
      error = (f_wb_i - y[i])
      for j in range(n):
        dj_dw[j] += error * X[i][j]
      dj_db += error
    dj_dw /= m
    dj_db /= m
    for j in range(n):
      dj_dw[j] += (lambda_ / m ) * w[j]
    return dj_dw, dj_db

  def gradient_descent(self, X, y, w_init, b_init, alpha, num_iters, lambda_ = 0):
    cost_history = []
    cost_iters = []
    w = copy.deepcopy(w_init)
    b = b_init
    for i in range(num_iters):
      dj_dw, dj_db = self.compute_gradient(X, y, w, b, lambda_)
      w = w - alpha * dj_dw
      b = b - alpha * dj_db
      if(i % (num_iters/10) == 0 or i == (num_iters - 1)):
        cost = self.compute_cost(X, y, w, b)
        cost_history.append(cost)
        cost_iters.append(i+1)
        print(f"Cost for iteration: {i+1}: {cost}")
    return cost_history, cost_iters, w, b
  
X = np.random.random(size = (50, 3))
y= [1 if (X[i][0] + X[i][1] + X[i][2]) > 1 else 0 for i in range(50) ]

w_initial = [0, 0, 0]
b_initial = 0
learning_rate = 0.01
number_of_iterations = 100
regressor = LogisticRegression()
cost_history, cost_iterations, w_final, b_final = regressor.gradient_descent(X, y, w_initial, b_initial, learning_rate, number_of_iterations)

y_pred = regressor.sigmoid(np.dot(X, w_final) + b_final)
