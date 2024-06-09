import numpy as np
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree

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
    
class AdaBoost2:
    def __init__(self, base_estimator, n_estimators=200):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.clfs = []
        self.alpha = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        weight = np.full(n_samples, 1 / n_samples)
        for _ in range(self.n_estimators):
            clf = self.base_estimator()
            if isinstance(clf, KNeighborsClassifier):
                clf.fit(X, y)
            else:
                clf.fit(X, y, sample_weight=weight)
            y_pred = clf.predict(X)
            self.clfs.append(clf)

            error_samples = (y != y_pred)
            error = np.sum(error_samples * weight)
            alpha = 0.5 * np.log((1 - error) / error)
            self.alpha.append(alpha)
            weight *= np.exp([alpha if error else -alpha for error in error_samples])
            weight /= np.sum(weight)

    def predict(self, X):
        y_pred = []
        for x in X:
            total = 0
            for i, clf in enumerate(self.clfs):
                prediction = clf.predict([x])[0]
                total += self.alpha[i] * (1 if prediction == 1 else -1)
            y_pred.append(1 if total >= 0 else 0)
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
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1-score:", f1_score(y_test, y_pred))
    
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    print("AUC:", auc)

    plt.plot(fpr, tpr, marker='.')
    plt.title('ROC Curve (Decision Tree)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()
    
    # The plot tree func need to use the class in sklearn.
    
    tree = DecisionTreeClassifier(criterion='gini', max_depth=7)
    tree.fit(X_train, y_train)
    
    plt.figure(figsize=(20,10))
    plot_tree(tree, filled=True, feature_names=train_df.columns[:-1], class_names=["Class 0", "Class 1"])
    plt.title("Decision Tree")
    plt.show()

    feature_importances = tree.feature_importances_
    features = train_df.columns[:-1]
    indices = np.argsort(feature_importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(X_train.shape[1]), feature_importances[indices], align="center")
    plt.xticks(range(X_train.shape[1]), features[indices], rotation=90)
    plt.xlim([-1, X_train.shape[1]])
    plt.show()

    for f in range(X_train.shape[1]):
        print(f"{features[indices[f]]}: {feature_importances[indices[f]]:.4f}")
    
    print("------------------------------------------------------------------")
    
    # Bayesian classification
    model1 = GaussianNB()
    model1.fit(X_train, y_train)
    model2 = MultinomialNB()
    model2.fit(X_train, y_train)
    model3 = BernoulliNB()
    model3.fit(X_train, y_train)
    
    print("GaussianNB accuracy:", model1.score(X_test, y_test))
    print("MultinomialNB accuracy:", model2.score(X_test, y_test))
    print("BernoulliNB accuracy:", model3.score(X_test, y_test))
    
    print("------------------------------------------------------------------")
    print("Below are from GaussianNB.")
    
    y_pred = model1.predict(X_test)
    
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1-score:", f1_score(y_test, y_pred))
    
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    print("AUC:", auc)

    plt.plot(fpr, tpr, marker='.')
    plt.title('ROC Curve (GaussianNB)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()
    
    print("------------------------------------------------------------------")
    
    # KNN
    k_range = range(2, 15)
    k_scores = []
    
    # kfold
    kf = StratifiedKFold(n_splits=10)
    
    for k in k_range:
        knn_model = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn_model, X_train, y_train, cv=kf, scoring="accuracy")
        print(f"K={k} Accuracy= {scores.mean()}")
        k_scores.append(scores.mean())

    best_k = k_scores.index(max(k_scores)) + 2
    print("Best K:", best_k)
    
    knn_model = KNeighborsClassifier(n_neighbors=best_k)
    knn_model.fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)
    
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1-score:", f1_score(y_test, y_pred))
    
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    print("AUC:", auc)

    plt.plot(fpr, tpr, marker='.')
    plt.title('ROC Curve (GaussianNB)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()
    
    print("------------------------------------------------------------------")
    
    # Adaboost
    print("AdaBoost with Decision Tree")
    ada = AdaBoost(criterion='gini', n_estimators=7)
    ada.fit(X_train, y_train)
    y_pred = ada.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    
    print("------------------------------------------------------------------")
    
    print("Adaboost with GaussianNB")
    ada_gnb = AdaBoost2(base_estimator=GaussianNB, n_estimators=50)
    ada_gnb.fit(X_train, y_train)
    y_pred = ada_gnb.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    
    print("------------------------------------------------------------------")

    print("Adaboost with KNN (Best K)")
    ada_knn = AdaBoost2(base_estimator=lambda: KNeighborsClassifier(n_neighbors=best_k), n_estimators=50)
    ada_knn.fit(X_train, y_train)
    y_pred = ada_knn.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    
    print("------------------------------------------------------------------")
    
    param_grid = {
        'n_neighbors': range(2, 15),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=10, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    print("Best parameters found by Grid Search:", grid_search.best_params_)
    
    param_dist = {
        'n_neighbors': range(2, 15),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }
    random_search = RandomizedSearchCV(KNeighborsClassifier(), param_dist, n_iter=20, cv=10, scoring='accuracy', random_state=0)
    random_search.fit(X_train, y_train)
    print("Best parameters found by Random Search:", random_search.best_params_)
    
    print("------------------------------------------------------------------")
    
    knn_model = grid_search.best_estimator_
    y_pred = knn_model.predict(X_test)

    print("KNN accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1-score:", f1_score(y_test, y_pred))

    fpr, tpr, _ = roc_curve(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    print("AUC:", auc)

    plt.plot(fpr, tpr, marker='.')
    plt.title('ROC Curve (KNN -- Grid)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()

    print("------------------------------------------------------------------")
    
    knn_model = random_search.best_estimator_
    y_pred = knn_model.predict(X_test)

    print("KNN accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1-score:", f1_score(y_test, y_pred))

    fpr, tpr, _ = roc_curve(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    print("AUC:", auc)

    plt.plot(fpr, tpr, marker='.')
    plt.title('ROC Curve (KNN -- Random)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()
