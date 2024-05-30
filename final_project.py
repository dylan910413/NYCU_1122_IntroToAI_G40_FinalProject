# You are not allowed to import any additional packages/libraries.
import numpy as np
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# This function computes the gini impurity of a label array.
def gini(y):
    unique_classes, counts = np.unique(y, return_counts=True)
    total_samples = len(y)
    gini_impurity = 1
    for count in counts:
        proportion = count / total_samples
        gini_impurity -= proportion ** 2
    return gini_impurity

# This function computes the entropy of a label array.
def entropy(y):
    unique_classes, counts = np.unique(y, return_counts=True)
    total_samples = len(y)
    entropy_val = 0
    for count in counts:
        proportion = count / total_samples
        entropy_val -= proportion * np.log2(proportion)
    return entropy_val
        
# The decision tree classifier class.
# Tips: You may need another node class and build the decision tree recursively.
class Node:
    def __init__(self, feature_index=None, split_value=None, left=None, right=None, value=None):
        self.feature_index = feature_index  # Feature index for splitting
        self.split_value = split_value      # Value used for splitting
        self.left = left                    # Left child node
        self.right = right                  # Right child node
        self.value = value                  # Value if the node is a leaf node (class prediction)
    def is_leaf_node(self):
        return self.value is not None
    
class DecisionTree():
    def __init__(self, criterion='gini', max_depth=None):
        self.criterion = criterion
        self.max_depth = max_depth 
        self.tree = None
    
    # This function computes the impurity based on the criterion.
    def impurity(self, y):
        if self.criterion == 'gini':
            return gini(y)
        elif self.criterion == 'entropy':
            return entropy(y)
    
    # This function fits the given data using the decision tree algorithm.
    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)
    
    # This function takes the input data X and predicts the class label y according to your trained model.
    def predict(self, X):
        return [self._traverse_tree(x, self.tree) for x in X]
    
    # This function plots the feature importance of the decision tree.
    def plot_feature_importance_img(self, columns):
        feature_usage = np.zeros(len(columns))
        self._count_feature_usage(self.tree, feature_usage)
        
        plt.figure(figsize=(10, 6))
        plt.barh(columns, feature_usage)
        plt.xlabel('Feature Usage')
        plt.title('Decision Tree Feature Importance')
        plt.show()

    def _count_feature_usage(self, node, feature_usage):
        if node is None:
            return
        
        if node.feature_index is not None:
            feature_usage[node.feature_index] += 1
        
        self._count_feature_usage(node.left, feature_usage)
        self._count_feature_usage(node.right, feature_usage)

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        if depth >= self.max_depth or n_classes == 1 or n_samples < 2:
            leaf_value = np.argmax(np.bincount(y))
            return Node(value=leaf_value)

        best_feature, best_threshold = None, None
        best_impurity = float('inf')

        for feature_idx in range(n_features):
            unique_values = np.unique(X[:, feature_idx])
            for threshold in unique_values:
                left_indices = np.where(X[:, feature_idx] <= threshold)[0]
                right_indices = np.where(X[:, feature_idx] > threshold)[0]
                if len(left_indices) > 0 and len(right_indices) > 0:
                    left_impurity = self.impurity(y[left_indices])
                    right_impurity = self.impurity(y[right_indices])

                    total_impurity = (len(left_indices) * left_impurity + len(right_indices) * right_impurity) / n_samples

                    if total_impurity < best_impurity:
                        best_impurity = total_impurity
                        best_feature = feature_idx
                        best_threshold = threshold

        if best_feature is not None:
            left_idxs = np.where(X[:, best_feature] <= best_threshold)[0]
            right_idxs = np.where(X[:, best_feature] > best_threshold)[0]
            left_child = self._build_tree(X[left_idxs], y[left_idxs], depth + 1)
            right_child = self._build_tree(X[right_idxs], y[right_idxs], depth + 1)
            
            if left_child is None:
                left_child = Node(value=np.argmax(np.bincount(y[left_idxs])))
            if right_child is None:
                right_child = Node(value=np.argmax(np.bincount(y[right_idxs])))
            return Node(feature_index=best_feature, split_value=best_threshold, left=left_child, right=right_child)

        leaf_value = np.argmax(np.bincount(y))
        return Node(value=leaf_value)
    def _traverse_tree(self, sample, node):
        if node is None:  
            return None
        if node.is_leaf_node():
            return node.value
        if sample[node.feature_index] <= node.split_value:
            return self._traverse_tree(sample, node.left)
        else:
            return self._traverse_tree(sample, node.right)
        
        
# The AdaBoost classifier class.
# best = 250
class AdaBoost():
    def __init__(self, criterion='gini', n_estimators = 0):
        self.criterion = criterion 
        self.n_estimators = n_estimators
        self.models = []  
        self.model_weights = []
        self.sample_weights = []
    # This function fits the given data using the AdaBoost algorithm.
    # You need to create a decision tree classifier with max_depth = 1 in each iteration.
    def fit(self, X, y):
        n_samples = X.shape[0]
        self.sample_weights = np.ones(n_samples) / n_samples 
        for _ in range(self.n_estimators):
            sampled_indices = np.random.choice(n_samples, n_samples, replace=True, p=self.sample_weights)
            X_sampled = X[sampled_indices]
            y_sampled = y[sampled_indices]
            
            tree = DecisionTree(criterion=self.criterion, max_depth=1)
            tree.fit(X_sampled, y_sampled) 
            
            y_pred_sampled = np.array(tree.predict(X_sampled))
            incorrect = (y_pred_sampled != y)
            err = np.sum(self.sample_weights * (incorrect)) 
            alpha = 0.5 * np.log((1 - err) / max(err, 1e-10))
            
            self.sample_weights *= np.exp(alpha * incorrect)
            self.sample_weights /= np.sum(self.sample_weights)
            
            self.models.append(tree)
            self.model_weights.append(alpha)

    def predict(self, X):
        n_samples = len(X)
        total_alpha = 0
        final_predictions = np.zeros(n_samples)
        
        for i, model in enumerate(self.models):
            total_alpha += self.model_weights[i]
            predictions = np.array(model.predict(X))
            final_predictions += self.model_weights[i] * predictions

        final_predictions /= total_alpha 
        final_predictions = np.where(final_predictions > 0, 1, 0)

        return final_predictions
# Do not modify the main function architecture.
# You can only modify the value of the random seed and the the arguments of your Adaboost class.
if __name__ == "__main__":
# Data Loading
    train_df = DataFrame(read_csv("train.csv"))
    test_df = DataFrame(read_csv("test.csv"))
    X_train = train_df.drop(["target"], axis=1)
    y_train = train_df["target"]
    X_test = test_df.drop(["target"], axis=1)
    y_test = test_df["target"]

    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

# Set random seed to make sure you get the same result every time.
# You can change the random seed if you want to.
    np.random.seed(0)
# Decision Tree
    print("Part 1: Decision Tree")
    data = np.array([0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1])
    print(f"gini of [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1]: {gini(data)}")
    print(f"entropy of [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1]: {entropy(data)}")
    tree = DecisionTree(criterion='gini', max_depth=7)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    print("Accuracy (gini with max_depth=7):", accuracy_score(y_test, y_pred))
    tree = DecisionTree(criterion='entropy', max_depth=7)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    print("Accuracy (entropy with max_depth=7):", accuracy_score(y_test, y_pred))
    #columns = ["age","sex","cp","fbs","thalach","thal"]
    #tree.plot_feature_importance_img(columns)
# AdaBoost
    print("Part 2: AdaBoost")
    # Tune the arguments of AdaBoost to achieve higher accuracy than your Decision Tree.
    ada = AdaBoost(criterion='gini', n_estimators=250)
    ada.fit(X_train, y_train)
    y_pred = ada.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    


