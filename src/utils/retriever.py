import os, pickle
import numpy as np
import faiss
import openai

INDEX_PATH = os.path.join(os.getcwd(), "faiss.index")
META_PATH  = os.path.join(os.getcwd(), "faiss_meta.pkl")
_index, _meta = None, None

def _load_store():
    global _index, _meta
    if _index is None:
        if not os.path.exists(INDEX_PATH):
            raise FileNotFoundError(f"{INDEX_PATH} not found")
        _index = faiss.read_index(INDEX_PATH)
    if _meta is None:
        if not os.path.exists(META_PATH):
            raise FileNotFoundError(f"{META_PATH} not found")
        with open(META_PATH, "rb") as f:
            _meta = pickle.load(f)
    return _index, _meta

def _embed_text(text: str) -> np.ndarray:
    resp = openai.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    emb = resp.data[0].embedding
    return np.array(emb, dtype="float32")

def get_top_k(query: str, k: int = 5) -> list[dict]:
    if not openai.api_key:
        raise RuntimeError("Set OPENAI_API_KEY before querying.")

    index, meta = _load_store()
    qvec = _embed_text(query)
    D, I = index.search(np.stack([qvec]), k)

    snippets = []
    for dist, idx in zip(D[0], I[0]):
        m = meta[idx]
        snippets.append({
            "text":   m["text"],
            "source": m["source"],
            "score":  float(dist),
        })
    return snippets
