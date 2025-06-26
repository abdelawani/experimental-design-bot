# Experimental Design Bot

A Streamlit-powered RAG chatbot that helps students of Dr. Abdelaziz Lawani’s Experimental Design course explore lecture notes, assignments, and textbook content. Under the hood it:

1. **Indexes** all your PDFs & DOCX via OpenAI embeddings + FAISS  
2. **Retrieves** the most relevant text chunks for each question  
3. **Answers** with GPT-4 Turbo, citing your sources  

---

## 📁 Repository Structure

experimental-design-bot/
├─ assets/ # images & logos
│ └ social_scientist.PNG
├─ data/ # your course materials
│ ├ assignments/ # .docx files
│ └ lectures/ # .pdf files
├─ src/
│ ├ init.py
│ └ utils/
│ ├ loader.py # build FAISS + metadata
│ ├ retriever.py # query FAISS + embed
│ └ formatter.py # Markdown formatting
├─ .env # your OpenAI key (git-ignored)
├─ .env.example # template for collaborators
├─ .gitattributes # LFS tracking of index files
├─ .gitignore
├─ faiss.index # stored in LFS
├─ faiss_meta.pkl # stored in LFS
├─ index.py # CLI entrypoint to (re)build index
├─ prompts.json # system-prompt for the chat
├─ requirements.txt
└─ streamlit_app.py # the Streamlit web UI


---

## ⚙️ Getting Started

### 1. Clone & enter project

```bash
git clone https://github.com/you/experimental-design-bot.git
cd experimental-design-bot

python -m pip install --upgrade pip
pip install -r requirements.txt


###2. Install Python dependencies

python -m pip install --upgrade pip
pip install -r requirements.txt

###3. Configure your OpenAI key
Copy .env.example → .env

Open .env, paste your key:

OPENAI_API_KEY=sk-…

###4. Build the FAISS index
This will read every .pdf/.docx in data/, generate embeddings, and write faiss.index + faiss_meta.pkl.

python index.py
You should see a message like:

Built 2102-vector FAISS index → ./faiss.index, ./faiss_meta.pkl

###5. Run the Streamlit app
streamlit run streamlit_app.py --server.port 8502
Then open your browser at http://localhost:8502.

### Deploying on Streamlit Cloud
Push your repo (with LFS-tracked index files) to GitHub.

On Streamlit Cloud, “New app” → link your GitHub repo → branch main → run streamlit_app.py.

Set the OPENAI_API_KEY secret in Streamlit’s dashboard.

Click “Deploy” and share the resulting public URL!

###Usage
Ask any Experimental Design question in the box

Cite function sources appear under each answer

Clear chat on the sidebar to restart

###License
MIT © Dr. Abdelaziz Lawani

