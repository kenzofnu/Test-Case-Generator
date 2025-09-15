import os
import requests
import tempfile
from dotenv import load_dotenv

# --- LangChain Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader

# --- Load API Keys and Constants ---
load_dotenv()

def load_api_keys():
    """Loads required API keys from .env file and returns them."""
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    shortcut_api_token = os.environ.get("SHORTCUT_API_TOKEN")
    shortcut_base_url = os.environ.get("SHORTCUT_API_BASE_URL", "https://api.app.shortcut.com/api/v3/stories/") # Default value

    if not google_api_key or not shortcut_api_token:
        raise ValueError("Please provide GOOGLE_API_KEY and SHORTCUT_API_TOKEN in your .env file.")
    return google_api_key, shortcut_api_token, shortcut_base_url

_, SHORTCUT_API_TOKEN, SHORTCUT_BASE_URL = load_api_keys()
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_DB_PATH = "./chroma_db"


# --- Core Functions ---

def get_embedding_function():
    """Returns the embedding function."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

def get_shortcut_story_details(story_id: int, base_url: str = None) -> str:
    """Fetches a story from the Shortcut API and formats it for the LLM."""
    url_to_use = base_url if base_url else SHORTCUT_BASE_URL
    if not url_to_use.endswith('/'):
        url_to_use += '/'
        
    full_url = f"{url_to_use}{story_id}"
    
    headers = {"Content-Type": "application/json", "Shortcut-Token": SHORTCUT_API_TOKEN}
    try:
        response = requests.get(full_url, headers=headers)
        response.raise_for_status()
        story_data = response.json()
        story_title = story_data.get("name", "No Title")
        story_description = story_data.get("description", "No Description")
        return f"**Feature:** {story_title}\n\n{story_description}"
    except requests.exceptions.RequestException as e:
        # Instead of st.error, which is Streamlit-specific, we'll return an error string.
        return f"Error: Could not fetch story from Shortcut. Details: {e}"

def get_retriever_from_files(uploaded_files, use_persistent_db=False):
    """Creates a vector store and retriever from uploaded file objects."""
    docs = []
    
    # Mapping file extensions to their corresponding loaders
    loader_map = {
        ".pdf": PyPDFLoader,
        ".docx": UnstructuredWordDocumentLoader,
        ".doc": UnstructuredWordDocumentLoader,
        ".txt": TextLoader,
        ".md": UnstructuredMarkdownLoader,
    }

    for uploaded_file in uploaded_files:
        # Use the file's name to determine the correct loader
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        loader_cls = loader_map.get(file_extension)

        if not loader_cls:
            # Skip files with unsupported extensions
            continue

        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            loader = loader_cls(tmp_file_path)
            docs.extend(loader.load())
        finally:
            os.remove(tmp_file_path)

    if not docs:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    doc_splits = text_splitter.split_documents(docs)

    embedding_function = get_embedding_function()
    
    if use_persistent_db:
        vectorstore = Chroma(
            persist_directory=CHROMA_DB_PATH, 
            embedding_function=embedding_function
        )
        vectorstore.add_documents(doc_splits)
    else:
        vectorstore = Chroma.from_documents(
            documents=doc_splits, 
            embedding=embedding_function
        )
        
    return vectorstore.as_retriever()


def get_retriever_from_directory(directory_path, use_persistent_db=True):
    """Creates a vector store and retriever from a local directory."""
    
    loader_map = {
        ".pdf": PyPDFLoader,
        ".docx": UnstructuredWordDocumentLoader,
        ".doc": UnstructuredWordDocumentLoader,
        ".txt": TextLoader,
        ".md": UnstructuredMarkdownLoader,
    }
    
    docs = []
    for ext, loader_cls in loader_map.items():
        loader = DirectoryLoader(directory_path, glob=f"**/*{ext}", loader_cls=loader_cls, silent_errors=True)
        docs.extend(loader.load())

    if not docs:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    doc_splits = text_splitter.split_documents(docs)

    embedding_function = get_embedding_function()
    
    db_path = CHROMA_DB_PATH if use_persistent_db else None
    
    if use_persistent_db and os.path.exists(CHROMA_DB_PATH):
        # Load existing DB and add new documents
        vectorstore = Chroma(
            persist_directory=db_path, 
            embedding_function=embedding_function
        )
        vectorstore.add_documents(doc_splits)
    else:
        # Create a new DB
        vectorstore = Chroma.from_documents(
            documents=doc_splits, 
            embedding=embedding_function,
            persist_directory=db_path
        )
        
    return vectorstore.as_retriever()


def create_rag_chain(llm, retriever, prompt_template):
    """Creates and returns a RAG chain."""
    document_chain = create_stuff_documents_chain(llm, prompt_template)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain
