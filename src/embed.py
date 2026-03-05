# embed.py
import faiss
import numpy as np
import os
import pickle
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_DIR = "embeddings"

# Load model once
model = SentenceTransformer(MODEL_NAME)


def embed_texts(texts, batch_size=64):
    """
    Convert texts to embeddings.
    Uses batching for speed.
    """
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return embeddings.astype("float32")


def build_and_save_index(texts, index_dir=INDEX_DIR):

    os.makedirs(index_dir, exist_ok=True)

    embeddings = embed_texts(texts)
    dim = embeddings.shape[1]

    # Cosine similarity index (better for normalized embeddings)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, os.path.join(index_dir, "faiss.index"))

    with open(os.path.join(index_dir, "texts.pkl"), "wb") as f:
        pickle.dump(texts, f)

    print(f"FAISS index built with {len(texts)} incidents")

    return index


def load_index(index_dir=INDEX_DIR):

    index_path = os.path.join(index_dir, "faiss.index")

    if not os.path.exists(index_path):
        raise ValueError("FAISS index not found. Run build_and_save_index first.")

    index = faiss.read_index(index_path)

    with open(os.path.join(index_dir, "texts.pkl"), "rb") as f:
        texts = pickle.load(f)

    return index, texts