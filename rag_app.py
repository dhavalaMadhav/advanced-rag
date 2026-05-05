import os
from dotenv import load_dotenv

print("🚀 Starting RAG system...")

# -----------------------------
# Load API Key
# -----------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# -----------------------------
# FastAPI setup
# -----------------------------
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Allow frontend requests
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
from sentence_transformers import SentenceTransformer

# -----------------------------
# Embedding Manager
# -----------------------------
class EmbeddingManager:
    def __init__(self):
        print("🧠 Loading embedding model (only once)...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()


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
# Create or Load DB
# -----------------------------
def get_vectorstore(embedding_manager):
    persist_dir = "./chroma_db"

    if os.path.exists(persist_dir):
        print("⚡ Loading existing DB (FAST START)")
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding_manager
        )

    print("🆕 Creating DB (first time setup)...")

    docs = load_documents("data")
    chunks = split_documents(docs)

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_manager,
        persist_directory=persist_dir
    )

    return db


# -----------------------------
# LOAD EVERYTHING ONCE (IMPORTANT)
# -----------------------------
embedding_manager = EmbeddingManager()
db = get_vectorstore(embedding_manager)
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
If the answer is not present, say:
"I don't know based on the given documents."

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
# Optional test route
# -----------------------------
@app.get("/")
def home():
    return {"message": "RAG API is running"}