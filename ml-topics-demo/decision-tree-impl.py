import numpy as np
class DecisionTree:

    tree = []
    
    def entropy(self, y):
        if(len(y) == 0 or sum(y) == 0 or sum(y) == len(y)):
            return 0
        else:
            p = sum(y) / len(y)
            return -p * (np.log2(p)) - (1 - p) * (np.log2(1 - p))
    
    def weighted_entropy(self, w1, p1, w2, p2):
        return w1 * self.entropy(p1) + w2 * self.entropy(p2)
    
    def information_gain(self, X, y, node_indices, feature):
        left_indices, right_indices = self.split_dataset(X, node_indices, feature)
        X_node, y_node = X[node_indices], y[node_indices]
        X_left, y_left = X[left_indices], y[left_indices]
        X_right, y_right = X[right_indices], y[right_indices]
        entropy_node = self.entropy(y_node)
        entropy_left = self.entropy(y_left)
        entropy_right = self.entropy(y_right)
        w_left = len(y_left) / len(y_node)
        w_right = 1 - w_left
        return (entropy_node - (w_left * entropy_left + w_right * entropy_right))
    
    def get_best_split(self, X, y, node_indices):
        num_features = X.shape[1]
        best_feature = -1
        max_info_gain = 0
        for i in range(num_features):
            info_gain = self.information_gain(X, y, node_indices, i)
            if(info_gain > max_info_gain):
                max_info_gain = info_gain
                best_feature = i
        return best_feature


    def split_dataset(self, X, node_indices, feature):
        left_indices = []
        right_indices = []
        for i in node_indices:
            if(X[i][feature] == 0):
                left_indices.append(i)
            else:
                right_indices.append(i)
        return left_indices, right_indices
    
    def build(self, X, y, node_indices, branch_name, max_depth, current_depth):
        if(current_depth == max_depth):
            print(f"Current Depth: {current_depth}")
            print(f"-- Branch: {branch_name},  Leaf Nodes: {node_indices}")
            return
        best_feature = self.get_best_split(X, y, node_indices)
        left_indices, right_indices = self.split_dataset(X, node_indices, best_feature)
        print(f"Current Depth: {current_depth}")
        print(f"-- Branch: {branch_name},  Splitting Feature: {best_feature}")
        self.tree.append((best_feature, left_indices, right_indices))
        self.build(X, y, left_indices, "Left", max_depth, current_depth + 1)
        self.build(X, y, right_indices, "Right", max_depth, current_depth + 1)

# Binary Features = [ear_shape, whiskers, face_shape], Target = [cat, dog]
X = np.array([[1,1,1],[1,0,1],[1,0,0],[1,0,0],[1,1,1],[0,1,1],[0,0,0],[1,0,1],[0,1,0],[1,0,0]])
y = np.array([1,1,0,0,1,0,0,1,1,0])
tree = DecisionTree()
tree.build(X, y, range(len(X)), "Root", max_depth=2, current_depth=0)