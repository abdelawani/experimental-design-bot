# streamlit_app.py

import os
from dotenv import load_dotenv
load_dotenv()  # 1) read .env so os.getenv("OPENAI_API_KEY") works

import json
import pickle
import streamlit as st
import openai
import faiss
from streamlit import chat_message

# 2) now import your utils from src/utils/
from src.utils.retriever import get_top_k
from src.utils.formatter import format_response

# ————————————————————————————————
#  CONFIGURE PATHS
# ————————————————————————————————
ROOT_DIR    = os.path.abspath(os.path.dirname(__file__))
PROMPT_PATH = os.path.join(ROOT_DIR, "prompts.json")
INDEX_PATH  = os.path.join(ROOT_DIR, "faiss.index")
META_PATH   = os.path.join(ROOT_DIR, "faiss_meta.pkl")
LOGO_PATH   = os.path.join(ROOT_DIR, "logo.png")

# ————————————————————————————————
#  PAGE CONFIG & BRANDING
# ————————————————————————————————
st.set_page_config(
    page_title="Experimental Design Bot",
    page_icon="🔬",
    layout="wide",
)

# ─── Your new “social scientist” illustration ────────────────────────
NEW_LOGO = os.path.join(ROOT_DIR, "assets", "social_scientist.png")
if os.path.exists(NEW_LOGO):
    st.image(NEW_LOGO, width=400)

# ─── Bot title + custom subtitle ────────────────────────────────────
st.title("🔬 Experimental Design Bot")
st.markdown(
    "*This chatbot is designed to assist students taking the Experimental Design course of Dr. Abdelaziz Lawani.*\n\n"
)

# ————————————————————————————————
#  SIDEBAR
# ————————————————————————————————
with st.sidebar:
    st.header("💡 How to use")
    st.write(
        "• Ask about experimental design topics\n"
        "• Latest answers appear first\n"
        "• Click “Clear chat” to restart"
    )
    if st.button("🗑️ Clear chat"):
        st.session_state.history = []
        st.session_state.input_key = 0

# ————————————————————————————————
#  LOAD PROMPTS
# ————————————————————————————————
if not os.path.isfile(PROMPT_PATH):
    st.error("❌ Missing prompts.json; run your indexer first.")
    st.stop()

with open(PROMPT_PATH) as f:
    PROMPTS = json.load(f)

# ————————————————————————————————
#  OPENAI KEY
# ————————————————————————————————
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    st.error("❌ Please set OPENAI_API_KEY in your environment (or .env).")
    st.stop()

# ————————————————————————————————
#  LOAD FAISS INDEX + METADATA
# ————————————————————————————————
@st.cache_resource
def get_store():
    if not (os.path.exists(INDEX_PATH) and os.path.exists(META_PATH)):
        st.error("❌ faiss.index / faiss_meta.pkl missing; re-run index.py")
        st.stop()
    idx = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    return {"index": idx, "meta": meta}

# ————————————————————————————————
#  CHAT HISTORY INITIALIZATION
# ————————————————————————————————
if "history" not in st.session_state:
    st.session_state.history = [
        {
            "query": None,
            "answer": "🤖🔬 Hi! I’m your Assistant for the Experimental Design course. Ask me anything about the materials discussed in class.",
            "sources": []
        }
    ]
if "input_key" not in st.session_state:
    st.session_state.input_key = 0

# ————————————————————————————————
#  QUERY HANDLER
# ————————————————————————————————
def handle_submit():
    key   = f"query_{st.session_state.input_key}"
    query = st.session_state.get(key, "").strip()
    if not query:
        return

    # 1) retrieve
        # 1) Retrieve top-k snippets
    try:
        get_store()                 # verifies index & meta are present
        snippets = get_top_k(query, k=5)

    except Exception as e:
        st.error(f"❌ Retrieval error: {e}")
        return

    # 2) build messages
    msgs = [{"role":"system","content":PROMPTS["system"]}]
    for s in snippets:
        msgs.append({"role":"system","content": s["text"]})
    msgs.append({"role":"user","content": query})

    # 3) call LLM
    try:
        with st.spinner("🤖 Thinking…"):
            resp = openai.chat.completions.create(
                model="gpt-4-turbo",
                messages=msgs,
                max_tokens=500
            )
    except Exception as e:
        st.error(f"❌ OpenAI API error: {e}")
        return

    raw   = resp.choices[0].message.content
    ans   = format_response(raw, snippets)

    # 4) dedupe sources
    urls = list(dict.fromkeys(s["source"] for s in snippets))

    # 5) append & bump
    st.session_state.history.append({
        "query":   query,
        "answer":  ans,
        "sources": urls
    })
    st.session_state.input_key += 1

# ————————————————————————————————
#  USER INPUT BOX
# ————————————————————————————————
st.text_input(
    "Ask me about experimental design…",
    key=f"query_{st.session_state.input_key}",
    placeholder="Type your question and press Enter",
    on_change=handle_submit
)

# ————————————————————————————————
#  RENDER CHAT — newest first
# ————————————————————————————————
RECENT = 5
hist   = st.session_state.history
older, recent = (hist[:-RECENT], hist[-RECENT:]) if len(hist) > RECENT else ([], hist)

for entry in reversed(recent):
    if entry["query"] is not None:
        with chat_message("user"):
            st.markdown(entry["query"])
    with chat_message("assistant"):
        st.markdown(entry["answer"])
        if entry["sources"]:
            with st.expander(f"🔗 Sources ({len(entry['sources'])})"):
                for u in entry["sources"]:
                    st.markdown(f"- {u}")
    st.markdown("---")

if older:
    with st.expander("📜 Show older messages"):
        for entry in reversed(older):
            if entry["query"] is not None:
                with chat_message("user"):
                    st.markdown(entry["query"])
            with chat_message("assistant"):
                st.markdown(entry["answer"])
                if entry["sources"]:
                    st.markdown("**Sources:**")
                    for u in entry["sources"]:
                        st.markdown(f"- {u}")
            st.markdown("---")
