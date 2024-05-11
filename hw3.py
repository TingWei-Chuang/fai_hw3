from collections import Counter

import numpy as np
import pandas as pd

# set random seed
np.random.seed(0)

"""
Tips for debugging:
- Use `print` to check the shape of your data. Shape mismatch is a common error.
- Use `ipdb` to debug your code
    - `ipdb.set_trace()` to set breakpoints and check the values of your variables in interactive mode
    - `python -m ipdb -c continue hw3.py` to run the entire script in debug mode. Once the script is paused, you can use `n` to step through the code line by line.
"""


# 1. Load datasets
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    DO NOT MODIFY THIS FUNCTION.
    """
    # Load iris dataset
    iris = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
        header=None,
    )
    iris.columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "class",
    ]

    # Load Boston housing dataset
    boston = pd.read_csv(
        "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
    )

    return iris, boston


# 2. Preprocessing functions
def train_test_split(
    df: pd.DataFrame, target: str, test_size: float = 0.3
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Shuffle and split dataset into train and test sets
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_size))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    # Split target and features
    X_train = train.drop(target, axis=1).values
    y_train = train[target].values
    X_test = test.drop(target, axis=1).values
    y_test = test[target].values

    return X_train, X_test, y_train, y_test


def normalize(X: np.ndarray) -> np.ndarray:
    # Normalize features to [0, 1]
    # You can try other normalization methods, e.g., z-score, etc.
    # TODO: 1%
    minimum = X.min(axis=0, keepdims=True)
    maximum = X.max(axis=0, keepdims=True)
    return (X - minimum) / (maximum - minimum)


def encode_labels(y: np.ndarray) -> np.ndarray:
    """
    Encode labels to integers.
    """
    # TODO: 1%
    codes, _ = pd.factorize(y, sort=True)
    return codes


# 3. Models
class LinearModel:
    def __init__(
        self, learning_rate=0.01, iterations=1000, model_type="linear"
    ) -> None:
        self.learning_rate = learning_rate
        self.iterations = iterations
        # You can try different learning rate and iterations
        self.model_type = model_type

        assert model_type in [
            "linear",
            "logistic",
        ], "model_type must be either 'linear' or 'logistic'"

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.insert(X, 0, 1, axis=1)
        n_classes = len(np.unique(y))
        n_features = X.shape[1]

        if self.model_type == "logistic":
            # one-hot encoded Y
            Y = np.zeros((X.shape[0], n_classes))
            Y[np.arange(X.shape[0]), y] = 1
            # initialize weight
            self.W = np.random.normal(0., 1., (n_features, n_classes))
            for epoch in range(self.iterations):
                # gradient descent
                self.W = self.W - self.learning_rate * self._compute_gradients(X, Y)
        else:
            Y = y[..., np.newaxis]
            # initiaize weight
            self.W = np.random.normal(0., 1., (n_features, 1))
            for epoch in range(self.iterations):
                # gradient descent
                self.W = self.W - self.learning_rate * self._compute_gradients(X, Y)
        # TODO: 2%

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.insert(X, 0, 1, axis=1)
        if self.model_type == "linear":
            # TODO: 2%
            Y_pred = X.dot(self.W).reshape((-1, ))
            return Y_pred
        elif self.model_type == "logistic":
            # TODO: 2%
            Z = X.dot(self.W)
            S = self._softmax(Z)
            Y_pred = S.argmax(axis=1, keepdims=False).astype(np.int64)
            return Y_pred

    def _compute_gradients(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.model_type == "linear":
            # TODO: 3%
            Z = X.dot(self.W)
            DL_DW = X.T.dot(Z - y) * 2 / X.shape[0]
            return DL_DW
        elif self.model_type == "logistic":
            # TODO: 3%
            Z = X.dot(self.W)
            S = self._softmax(Z)
            DL_DZ = S - y
            DL_DW = X.T.dot(DL_DZ) # derivatives
            return DL_DW

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        exp = np.exp(z)
        return exp / np.sum(exp, axis=1, keepdims=True)


class DecisionTree:
    def __init__(self, max_depth: int = 5, model_type: str = "classifier"):
        self.max_depth = max_depth
        self.model_type = model_type

        assert model_type in [
            "classifier",
            "regressor",
        ], "model_type must be either 'classifier' or 'regressor'"

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.tree = self._build_tree(X, y, 0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> dict:
        if depth >= self.max_depth or self._is_pure(y):
            return self._create_leaf(y)

        feature, threshold = self._find_best_split(X, y)
        # TODO: 4%
        mask = X[:, feature] <= threshold
        X_left = X[mask, :]
        y_left = y[mask]
        X_right = X[~mask, :]
        y_right = y[~mask]
        left_child = self._build_tree(X_left, y_left, depth + 1)
        right_child = self._build_tree(X_right, y_right, depth + 1)
        return {
            "feature": feature,
            "threshold": threshold,
            "left": left_child,
            "right": right_child,
        }

    def _is_pure(self, y: np.ndarray) -> bool:
        return len(set(y)) == 1

    def _create_leaf(self, y: np.ndarray):
        if self.model_type == "classifier":
            # TODO: 1%
            votes = dict()
            for label in y:
                if votes.get(label) == None:
                    votes[label] = 1
                else:
                    votes[label] += 1
            maxval = 0
            selected = 0
            for key, value in votes.items():
                if value > maxval:
                    value = maxval
                    selected = key
            return selected
        else:
            # TODO: 1%
            return y.sum() / y.shape[0]

    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> tuple[int, float]:
        best_gini = float("inf")
        best_mse = float("inf")
        best_feature = None
        best_threshold = None

        for feature in range(X.shape[1]):
            sorted_indices = np.argsort(X[:, feature])
            for i in range(1, len(X)):
                if X[sorted_indices[i - 1], feature] != X[sorted_indices[i], feature]:
                    threshold = (
                        X[sorted_indices[i - 1], feature]
                        + X[sorted_indices[i], feature]
                    ) / 2
                    mask = X[:, feature] <= threshold
                    left_y, right_y = y[mask], y[~mask]

                    if self.model_type == "classifier":
                        gini = self._gini_index(left_y, right_y)
                        if gini < best_gini:
                            best_gini = gini
                            best_feature = feature
                            best_threshold = threshold
                    else:
                        mse = self._mse(left_y, right_y)
                        if mse < best_mse:
                            best_mse = mse
                            best_feature = feature
                            best_threshold = threshold

        return best_feature, best_threshold

    def _gini_index(self, left_y: np.ndarray, right_y: np.ndarray) -> float:
        # TODO: 4%
        num_left = left_y.shape[0]
        classes_left = list(set(left_y))
        gini_left = 1.
        for c in range(len(classes_left)):
            mask = left_y == classes_left[c]
            num = left_y[mask].shape[0]
            prob = 1. * num / num_left
            gini_left -= prob * prob
        num_right = right_y.shape[0]
        classes_right = list(set(right_y))
        gini_right = 1.
        for c in range(len(classes_right)):
            mask = right_y == classes_right[c]
            num = right_y[mask].shape[0]
            prob = 1. * num / num_right
            gini_right -= prob * prob
        proportion = 1. * left_y.shape[0] / (left_y.shape[0] + right_y.shape[0])
        return proportion * gini_left + (1 - proportion) * gini_right


    def _mse(self, left_y: np.ndarray, right_y: np.ndarray) -> float:
        # TODO: 4%
        mean_left = left_y.sum() / left_y.shape[0]
        mse_left = ((left_y - mean_left) ** 2).sum() / left_y.shape[0]
        mean_right = right_y.sum() / right_y.shape[0]
        mse_right = ((right_y - mean_right) ** 2).sum() / right_y.shape[0]
        proportion = 1. * left_y.shape[0] / (left_y.shape[0] + right_y.shape[0])
        return proportion * mse_left + (1 - proportion) * mse_right

    def _traverse_tree(self, x: np.ndarray, node: dict):
        if isinstance(node, dict):
            feature, threshold = node["feature"], node["threshold"]
            if x[feature] <= threshold:
                return self._traverse_tree(x, node["left"])
            else:
                return self._traverse_tree(x, node["right"])
        else:
            return node


class RandomForest:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        model_type: str = "classifier",
    ):
        # TODO: 1%
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model_type = model_type
        self.trees = [DecisionTree(max_depth, model_type) for _ in range(n_estimators)]

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        for tree in self.trees:
            # TODO: 2%
            # bootstrap_indices = np.random.choice(
            indices = np.random.choice(X.shape[0], int(X.shape[0] / 3))
            X_in = X[indices, :]
            y_in = y[indices]
            tree.fit(X_in, y_in)

    def predict(self, X: np.ndarray) -> np.ndarray:
        # TODO: 2%
        if self.model_type == "classifier":
            votes = [dict() for _ in range(X.shape[0])]
            for tree in self.trees:
                pred = tree.predict(X)
                for i, p in enumerate(pred):
                    if votes[i].get(p) == None:
                        votes[i][p] = 1
                    else:
                        votes[i][p] += 1
            y_pred = []
            for vote in votes:
                max_v = 0
                best_c = []
                for c, v in vote.items():
                    if v > max_v:
                        max_v = v
                        best_c = [c]
                    elif v == max_v:
                        best_c.append(c)
                y_pred.append(np.random.choice(best_c))
            return np.array(y_pred)
        else:
            arr = []
            for tree in self.trees:
                pred = tree.predict(X)
                arr.append(pred)
            arr = np.array(arr)
            return arr.mean(0, keepdims=False)


# 4. Evaluation metrics
def accuracy(y_true, y_pred):
    # TODO: 1%
    return (y_true == y_pred).sum() / y_true.shape[0]


def mean_squared_error(y_true, y_pred):
    # TODO: 1%
    a = y_true - y_pred
    return a.T.dot(a) / y_true.shape[0]


# 5. Main function
def main():
    iris, boston = load_data()

    # Iris dataset - Classification
    X_train, X_test, y_train, y_test = train_test_split(iris, "class")
    X_train, X_test = normalize(X_train), normalize(X_test)
    y_train, y_test = encode_labels(y_train), encode_labels(y_test)

    logistic_regression = LinearModel(model_type="logistic")
    logistic_regression.fit(X_train, y_train)
    y_pred = logistic_regression.predict(X_test)
    print("Logistic Regression Accuracy:", accuracy(y_test, y_pred))

    decision_tree_classifier = DecisionTree(model_type="classifier")
    decision_tree_classifier.fit(X_train, y_train)
    y_pred = decision_tree_classifier.predict(X_test)
    print("Decision Tree Classifier Accuracy:", accuracy(y_test, y_pred))

    random_forest_classifier = RandomForest(model_type="classifier")
    random_forest_classifier.fit(X_train, y_train)
    y_pred = random_forest_classifier.predict(X_test)
    print("Random Forest Classifier Accuracy:", accuracy(y_test, y_pred))

    # Boston dataset - Regression
    X_train, X_test, y_train, y_test = train_test_split(boston, "medv")
    X_train, X_test = normalize(X_train), normalize(X_test)

    linear_regression = LinearModel(model_type="linear")
    linear_regression.fit(X_train, y_train)
    y_pred = linear_regression.predict(X_test)
    print("Linear Regression MSE:", mean_squared_error(y_test, y_pred))

    decision_tree_regressor = DecisionTree(model_type="regressor")
    decision_tree_regressor.fit(X_train, y_train)
    y_pred = decision_tree_regressor.predict(X_test)
    print("Decision Tree Regressor MSE:", mean_squared_error(y_test, y_pred))

    random_forest_regressor = RandomForest(model_type="regressor")
    random_forest_regressor.fit(X_train, y_train)
    y_pred = random_forest_regressor.predict(X_test)
    print("Random Forest Regressor MSE:", mean_squared_error(y_test, y_pred))


if __name__ == "__main__":
    main()
