import os, pickle
from pathlib import Path

import numpy as np
import faiss
import openai
import tiktoken
from PyPDF2 import PdfReader
from docx import Document

# ─── CONFIG ─────────────────────────────────────────────────────────
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50
BASE_DIR      = Path(__file__).resolve().parents[2]
DATA_DIR      = BASE_DIR / "data"
INDEX_PATH    = BASE_DIR / "faiss.index"
META_PATH     = BASE_DIR / "faiss_meta.pkl"

# ─── HELPERS ────────────────────────────────────────────────────────
def read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def read_docx(path: Path) -> str:
    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)

def split_into_chunks(text: str) -> list[str]:
    enc    = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    chunks = []
    start  = 0
    while start < len(tokens):
        chunk_tok = tokens[start : start + CHUNK_SIZE]
        chunks.append(enc.decode(chunk_tok))
        start += (CHUNK_SIZE - CHUNK_OVERLAP)
    return chunks

# ─── MAIN INDEXER ───────────────────────────────────────────────────
def build_faiss_index():
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise RuntimeError("Set OPENAI_API_KEY before indexing.")

    metas, vectors = [], []

    for fp in DATA_DIR.rglob("*"):
        suf = fp.suffix.lower()
        if suf == ".pdf":
            text = read_pdf(fp)
        elif suf in (".docx", ".doc"):
            text = read_docx(fp)
        else:
            continue

        doc_id = fp.stem
        for idx, chunk in enumerate(split_into_chunks(text)):
            # ← use new API here
            resp = openai.embeddings.create(
                model="text-embedding-ada-002",
                input=chunk
            )
            emb = resp.data[0].embedding

            metas.append({
                "doc_id":      doc_id,
                "chunk_index": idx,
                "text":        chunk,
                "source":      str(fp.relative_to(BASE_DIR)),
            })
            vectors.append(emb)

    if not vectors:
        raise RuntimeError("No embeddings—check your data/ for .pdf/.docx files.")

    x   = np.array(vectors, dtype="float32")
    dim = x.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(x)

    faiss.write_index(index, str(INDEX_PATH))
    with open(META_PATH, "wb") as f:
        pickle.dump(metas, f)

    print(f"Built {len(vectors)}-vector FAISS index → {INDEX_PATH}, {META_PATH}")

# alias for index.py
load_documents = build_faiss_index
