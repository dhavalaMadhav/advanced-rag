import os
from dotenv import load_dotenv

print("🚀 Starting RAG system...")

# -----------------------------
# Load API Keys
# -----------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# -----------------------------
# FastAPI setup
# -----------------------------
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Imports
# -----------------------------
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
# -----------------------------
# Load PDFs
# -----------------------------
def load_documents(folder="data"):
    docs = []
    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            path = os.path.join(folder, file)
            loader = PyMuPDFLoader(path)
            loaded_docs = loader.load()

            for d in loaded_docs:
                d.metadata["source"] = file

            docs.extend(loaded_docs)
    return docs

# -----------------------------
# Split Documents
# -----------------------------
def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    return splitter.split_documents(docs)

# -----------------------------
# Create / Load DB
# -----------------------------
def get_vectorstore():
    # 🔥 Use persistent folder (important)
    persist_dir = "./chroma_db"

    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=HF_TOKEN,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # --------------------------------------------------
    # ❌ DISABLED: Heavy embedding (causes Render crash)
    # --------------------------------------------------

    # docs = load_documents("data")
    # chunks = split_documents(docs)
    #
    # db = Chroma.from_documents(
    #     documents=chunks,
    #     embedding=embeddings,
    #     persist_directory=persist_dir
    # )

    # --------------------------------------------------
    # ✅ ONLY LOAD EXISTING DB
    # --------------------------------------------------
    db = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )

    return db

# -----------------------------
# LOAD SYSTEM
# -----------------------------
print("📦 Loading vector DB...")
db = get_vectorstore()
retriever = db.as_retriever(search_kwargs={"k": 3})

print("🤖 Connecting to Groq...")
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant",
    temperature=0.1
)

print("✅ Backend ready!")

# -----------------------------
# Request model
# -----------------------------
class Query(BaseModel):
    question: str

# -----------------------------
# API Endpoint
# -----------------------------
@app.post("/ask")
def ask(query: Query):
    docs = retriever.invoke(query.question)

    if not docs:
        return {"answer": "No relevant documents found", "sources": []}

    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
Answer ONLY using the context below.
If not found, say: I don't know.

Context:
{context}

Question:
{query.question}
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "sources": [
            {
                "file": d.metadata.get("source"),
                "page": d.metadata.get("page")
            }
            for d in docs
        ]
    }

# -----------------------------
# Health check
# -----------------------------
@app.get("/")
def home():
    return {"message": "RAG API is running"}