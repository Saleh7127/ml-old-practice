import numpy as np
from collections import Counter, defaultdict
from scipy.spatial import distance


class KNN:
    def __init__(self, k=3, distance_metric='euclidean', weighted=False):
        self.k = k
        self.distance_metric = distance_metric
        self.weighted = weighted

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        X = np.array(X)
        return [self._predict(x) for x in X]

    def score(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)

    def _predict(self, x):
        # Compute distances
        distances = [self._compute_distance(x, x_train) for x_train in self.X_train]

        # Get indices of k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        if self.weighted:
            # Weighted vote
            label_weights = defaultdict(float)
            for i in k_indices:
                label = self.y_train[i]
                dist = distances[i]
                weight = 1 / (dist + 1e-5)  # avoid divide by zero
                label_weights[label] += weight
            return max(label_weights.items(), key=lambda x: x[1])[0]
        else:
            # Majority vote
            most_common = Counter(k_nearest_labels).most_common()
            top_count = most_common[0][1]
            top_labels = [label for label, count in most_common if count == top_count]

            if len(top_labels) == 1:
                return top_labels[0]
            else:
                # Tie-breaking: return class of closest point among tied
                for i in k_indices:
                    if self.y_train[i] in top_labels:
                        return self.y_train[i]

    def _compute_distance(self, x1, x2):
        if self.distance_metric == 'euclidean':
            return np.linalg.norm(x1 - x2)
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        elif self.distance_metric == 'cosine':
            return distance.cosine(x1, x2)
        elif self.distance_metric == 'minkowski':
            return distance.minkowski(x1, x2, p=3)
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")
