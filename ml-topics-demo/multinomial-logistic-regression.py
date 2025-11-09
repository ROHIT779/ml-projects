import numpy as np
class MultinomialLogisticRegression:

  def __init__(self, num_features, num_classes, learning_rate=0.01, epochs=1000):
      self.num_features = num_features
      self.num_classes = num_classes
      self.learning_rate = learning_rate
      self.epochs = epochs
      self.W = np.random.randn(num_features, num_classes) * 0.01
      self.b = np.zeros((1, num_classes))

  def softmax(self, z):
    z_max = np.max(z, axis = 1, keepdims = True)
    exp_z = np.exp(z - z_max)
    return exp_z / np.sum(exp_z, axis = 1, keepdims = True)

  def cross_entropy_loss(self, y, y_pred):
    m = y.shape[0]
    y_pred_log = np.log(y_pred + 1e-10)
    return -np.sum(y * y_pred_log) / m

  def one_hot_encode(self, y, num_classes):
    one_hot = np.zeros((y.size, num_classes))
    one_hot[np.arange(y.size), y - 1] = 1
    return one_hot

  def fit(self, X, y):
    m = X.shape[0]
    Y = self.one_hot_encode(y, self.num_classes)
    loss_history = []
    epoch_history = []
    for epoch in range(self.epochs):
      z = np.dot(X, self.W) + self.b
      y_pred = self.softmax(z)
      loss = self.cross_entropy_loss(Y, y_pred)
      if(epoch % 10 == 0):
        print(f"Epoch: {epoch},  Loss: {loss:.4f}")
        epoch_history.append(epoch)
        loss_history.append(loss)
      error = y_pred - Y
      dj_dw = np.dot(X.T, error) / m
      dj_db = np.sum(error, axis = 0, keepdims = True) / m
      self.W -= self.learning_rate * dj_dw
      self.b -= self.learning_rate * dj_db
    return epoch_history, loss_history

  def predict(self, X):
    z = np.dot(X, self.W) + self.b
    y_pred = self.softmax(z)
    return np.argmax(y_pred, axis = 1)

  def predict_prob(self, X):
    z = np.dot(X, self.W) + self.b
    return self.softmax(z)
  
X = np.random.random(size = (50, 3))
y = np.array([np.random.randint(1, 5) for i in range(50)])

num_features = 3
num_classes = 4
model = MultinomialLogisticRegression(num_features, num_classes, learning_rate=0.2, epochs=101)
epochs, losses = model.fit(X, y)