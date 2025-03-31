import streamlit as st
import os
from dotenv import load_dotenv
import pathlib

# === Load environment ===
load_dotenv()

# Use Streamlit secrets if available, otherwise fallback to .env
openai_api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
os.environ["OPENAI_API_KEY"] = openai_api_key  # Set globally for LlamaIndex

# === LlamaIndex imports ===
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.prompts import PromptTemplate
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.indices.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine

# === Streamlit page config ===
st.set_page_config(page_title="Coug GPT", page_icon="🎓")

# === Constants ===
PERSIST_DIR = "storage"
DATA_DIR = "data"

# === Set LLM and embedding model ===
Settings.embed_model = OpenAIEmbedding(
    api_key=openai_api_key,
    model="text-embedding-ada-002"
)

Settings.llm = OpenAI(
    api_key=openai_api_key,
    model="gpt-4o",
    temperature=0.3,
    system_prompt="""
    You are a professional university assistant that answers questions based on the given university PDF documents. 
    This GPT answers student questions pertaining to the student handbook, academic policies, 2024-2025 academic catalog, and financial aid handbook. 
    Every response must include specific page numbers and, where applicable, section references from these PDFs, based on the page numbers displayed in the PDF viewer. 
    Providing accurate, succinct, and well-cited answers is mandatory for every response.
    - Be concise and clear.
    - Only answer with information found in the PDF documents.
    - Do NOT hallucinate.
    """
)

# === Prompt template ===
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

# === Load or build index ===
@st.cache_resource
def load_index():
    if not os.path.exists(PERSIST_DIR):
        from llama_index.core.node_parser import SentenceSplitter
        documents = SimpleDirectoryReader(DATA_DIR).load_data()
        for doc in documents:
            path = doc.metadata.get("file_path", "")
            doc.metadata["file_name"] = os.path.basename(path)

        splitter = SentenceSplitter(chunk_size=200, chunk_overlap=50)
        nodes = splitter.get_nodes_from_documents(documents)

        index = VectorStoreIndex.from_documents(documents, show_progress=True)
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
    return index

index = load_index()

# === Retriever & query engine ===
retriever = VectorIndexRetriever(index=index, similarity_top_k=25)
postprocessor = SimilarityPostprocessor(similarity_cutoff=0.25)

query_engine = RetrieverQueryEngine.from_args(
    retriever=retriever,
    node_postprocessors=[postprocessor],
    text_qa_template=custom_template,
)

# === Streamlit UI ===
st.title("Welcome to Coug GPT")
st.markdown("CCU GPT can answer general questions from the student handbook, graduation documents, course catalog, academic policies, and financial aid handbook. Be as specific as possible when asking questions.")
st.image("logo.png", width=800)
st.markdown("**DISCLAIMER: This tool does not replace academic advising. For official questions, please contact your advisor.**")

user_query = st.text_input("Ask a question:")

if user_query:
    with st.spinner("Thinking..."):
        response = query_engine.query(user_query)

    st.markdown("### 💬 Answer")
    st.markdown(response.response)
