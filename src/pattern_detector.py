from sklearn.cluster import KMeans

class FailurePatternDetector:

    def __init__(self, embeddings):

        self.embeddings = embeddings

    def detect_patterns(self, n_clusters=8):

        kmeans = KMeans(n_clusters=n_clusters)

        labels = kmeans.fit_predict(self.embeddings)

        clusters = {}

        for i, label in enumerate(labels):

            clusters.setdefault(label, []).append(i)

        return clusters