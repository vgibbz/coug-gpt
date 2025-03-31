from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

# === Load environment ===
load_dotenv(override=True)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

from llama_index.core import (
    VectorStoreIndex, SimpleDirectoryReader, StorageContext,
    load_index_from_storage, Settings
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.prompts import PromptTemplate
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.indices.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine

# === App setup ===
app = FastAPI()

# Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Constants ===
PERSIST_DIR = "storage"
DATA_DIR = "data"

# === Embeddings & LLM setup ===
Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
Settings.llm = OpenAI(
    model="gpt-4o",
    temperature=0.3,
    system_prompt="""
    You are a professional university assistant that answers questions based on the given university PDF documents. 
    This GPT answers student questions pertaining to the student handbook, academic policies, 2024-2025 academic catalog, and financial aid handbook. 
    Every response must include specific page numbers and, where applicable, section references from these PDFs. Be concise and accurate.
    """
)

custom_template = PromptTemplate(
    "Use only the context below to answer the question.\n"
    "At the end of your response, cite your answer with the file name and page number if available, like (Catalog, page 12).\n"
    "If the answer is not in the context, say 'I don't know the answer.' Do not guess.\n\n"
    "Question: {query_str}\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Answer:"
)

# === Load or create index ===
def load_index():
    if not os.path.exists(PERSIST_DIR):
        documents = SimpleDirectoryReader(DATA_DIR).load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
    return index

index = load_index()

retriever = VectorIndexRetriever(index=index, similarity_top_k=25)
postprocessor = SimilarityPostprocessor(similarity_cutoff=0.25)
query_engine = RetrieverQueryEngine.from_args(
    retriever=retriever,
    node_postprocessors=[postprocessor],
    text_qa_template=custom_template,
)

# === Request & Response Models ===
class Question(BaseModel):
    question: str

@app.post("/ask")
def ask_question(q: Question):
    response = query_engine.query(q.question)
    return {"answer": response.response}
