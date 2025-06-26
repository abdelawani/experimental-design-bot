#!/usr/bin/env python
# index.py

import os
from dotenv import load_dotenv

# This import points at your build_faiss_index() in loader.py
from src.utils.loader import load_documents  

def main():
    # ─── 1) Load .env ───────────────────────────────
    load_dotenv()  

    # ─── 2) Grab & sanity-check your key ────────────
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Please set OPENAI_API_KEY in your .env or environment")
    import openai
    openai.api_key = key

    # ─── 3) Build your FAISS index & metadata ───────
    load_documents()  # this calls build_faiss_index()

if __name__ == "__main__":
    main()
