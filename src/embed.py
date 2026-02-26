import faiss
import numpy as np
import os
import pickle
from sentence_transformers import SentenceTransformer

# Load a pre-trained model once
model = SentenceTransformer("all-MiniLM-L6-v2")  # lightweight, fast model

def embed_texts(texts):
    """
    Converts a list of texts into embeddings using SentenceTransformer.
    """
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings.astype("float32")


def build_and_save_index(texts, index_dir="embeddings"):
    """
    Builds a FAISS index for a list of texts and saves it.
    """
    os.makedirs(index_dir, exist_ok=True)

    embeddings = embed_texts(texts)
    dim = embeddings.shape[1]

    # Create FAISS index (L2 distance)
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Save FAISS index and original texts
    faiss.write_index(index, os.path.join(index_dir, "faiss.index"))
    with open(os.path.join(index_dir, "texts.pkl"), "wb") as f:
        pickle.dump(texts, f)

    return index


def load_index(index_dir="embeddings"):
    """
    Loads the FAISS index and associated texts.
    """
    index = faiss.read_index(os.path.join(index_dir, "faiss.index"))
    with open(os.path.join(index_dir, "texts.pkl"), "rb") as f:
        texts = pickle.load(f)
    return index, texts