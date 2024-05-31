import numpy as np
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

def gini(y, weight):
    classes = np.unique(y)
    gini = 1
    if weight is None:
        for cls in classes:
            proportion = len(y[np.where(y == cls)]) / len(y)
            gini -= proportion**2
    else:
        total_weight = np.sum(np.array(weight))
        for cls in classes:
            proportion = np.sum(weight[np.where(y == cls)]) / total_weight
            gini -= proportion**2
    return gini

class Node():
    def __init__(self, feature=None, threshold=None, left=None, right=None, cls=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.cls = cls
        
class DecisionTree():
    def __init__(self, criterion='gini', max_depth=None):
        self.criterion = criterion
        self.max_depth = max_depth 
    
    def impurity(self, y, weight=None):
        return gini(y, weight)
    
    def fit(self, X, y, weight=None):
        y = y.reshape(-1, 1)
        self.feature_importance = np.zeros(X.shape[1])
        data = np.concatenate((X, y), axis=1)
        self.root = self.build_tree(data, 0, weight)
    
    def build_tree(self, data, depth, weight):
        X, y = data[:, :-1], data[:, -1]
        num_features = X.shape[1]
        if depth < self.max_depth:
            split = self.get_split(data, num_features, weight)
            if split['information_gain'] > 0:
                left_weight, right_weight = split['left_weight'], split['right_weight']
                left_subtree = self.build_tree(split['left_data'], depth + 1, left_weight)
                right_subtree = self.build_tree(split['right_data'], depth + 1, right_weight)
                self.feature_importance[split['feature']] += 1
                return Node(split['feature'], split['threshold'], left_subtree, right_subtree)

        y = list(y)
        cls = max(y, key=y.count)
        return Node(cls = cls)
    
    def get_split(self, data, num_features, weight):
        best_gain = {'feature': None, 'threshold': None, 'information_gain': 0,
                'left_data': None, 'right_data': None, 'left_weight': None, 'right_weight': None}
        max_gain = 0
        features = np.arange(num_features)
        for feature in features:
            col = data[:, feature]
            thresholds = np.unique(col)
            for threshold in thresholds:
                left_data , right_data = data[np.where(col <= threshold)], data[np.where(col > threshold)]
                if len(left_data) and len(right_data):
                    y_total, y_left, y_right = data[:, -1], left_data[:, -1], right_data[:, -1]
                    
                    if weight is None:
                        left_weight, right_weight = None, None
                        left_ratio, right_ratio = len(y_left) / len(y_total), len(y_right) / len(y_total)
                    else:
                        left_weight, right_weight = weight[np.where(col <= threshold)], weight[np.where(col > threshold)]
                        left_ratio, right_ratio = np.sum(left_weight) / np.sum(weight), np.sum(right_weight) / np.sum(weight)
                        
                    information_gain = self.impurity(y_total, weight) - (left_ratio * self.impurity(y_left, left_weight) + right_ratio * self.impurity(y_right, right_weight))
                    if information_gain > max_gain:
                        max_gain = information_gain
                        best_gain = {'feature': feature, 'threshold': threshold, 'information_gain': information_gain,
                                'left_data': left_data, 'right_data': right_data,
                                'left_weight': left_weight, 'right_weight': right_weight}
        return best_gain
    
    def predict(self, X):
        pred_y = []
        for x in X:
            pred_y.append(self.make_predict(x, self.root))
        return np.array(pred_y)
    
    def make_predict(self, x, node):
        if node.cls != None:
            return node.cls
        return self.make_predict(x, node.left) if x[node.feature] <= node.threshold else self.make_predict(x, node.right)

class AdaBoost():
    def __init__(self, criterion='gini', n_estimators=200):
        self.criterion = criterion 
        self.n_estimators = n_estimators
        self.clfs = []
        self.alpha = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        weight = np.full(n_samples, 1 / n_samples)
        for _ in range(self.n_estimators):
            clf = DecisionTree(max_depth = 1, criterion=self.criterion)
            clf.fit(X, y, weight)
            y_pred = clf.predict(X)
            self.clfs.append(clf)

            error_samples = (y != y_pred)
            error = np.sum(error_samples * weight)
#test 0.3 ~ 0.7 (0.3 = best)
            alpha = 0.3 * np.log((1 - error) / error)
            self.alpha.append(alpha)
            weight *= np.exp([alpha if error else - alpha for error in error_samples])
            weight /= np.sum(weight)

    def predict(self, X):
        y_pred = []
        for x in X:
            total = 0
            for i, clf in enumerate(self.clfs):
                prediction = clf.make_predict(x, clf.root)
                if prediction == 0:
                    total -= self.alpha[i]
                else:
                    total += self.alpha[i]
            y_pred.append(0 if total < 0 else 1)
        return np.array(y_pred)

if __name__ == "__main__":
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

    np.random.seed(0)
    print("Decision Tree")
    tree = DecisionTree(criterion='gini', max_depth=7)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    print("Accuracy (gini with max_depth=7):", accuracy_score(y_test, y_pred))
    
# Plotting the feature importance
    # columns = train_df.iloc[:, :-1].columns
    # tree.plot_feature_importance_img(columns)

# AdaBoost
    print("AdaBoost")
    ada = AdaBoost(criterion='gini', n_estimators=7)
    ada.fit(X_train, y_train)
    y_pred = ada.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))