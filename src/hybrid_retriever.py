from rank_bm25 import BM25Okapi
from embed import embed_texts, load_index

class HybridRetriever:

    def __init__(self):

        self.index, self.texts = load_index()

        tokenized = [t.split() for t in self.texts]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query, k=3):

        # BM25 search
        tokenized_query = query.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)

        bm25_top = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True
        )[:k]

        # FAISS search
        query_vec = embed_texts([query])
        _, faiss_idx = self.index.search(query_vec, k)

        combined = list(set(bm25_top + list(faiss_idx[0])))

        results = [self.texts[i][:400] for i in combined[:k]]

        return results