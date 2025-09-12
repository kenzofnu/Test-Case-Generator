import os
import requests # <-- NEW: Import the requests library
from dotenv import load_dotenv

# --- Part 1: Import all necessary libraries (mostly the same) ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Part 2: Load API Keys and Set Up Paths/Constants ---
load_dotenv()

# Load Google Key
if "GOOGLE_API_KEY" not in os.environ:
    raise ValueError("Google API key not found in .env file.")

# --- NEW: Load Shortcut Key and set constants ---
if "SHORTCUT_API_TOKEN" not in os.environ:
    raise ValueError("Shortcut API token not found in .env file.")
SHORTCUT_API_TOKEN = os.environ["SHORTCUT_API_TOKEN"]
SHORTCUT_BASE_URL = "https://api.app.shortcut.com/api/v3/stories/"
# --- END NEW ---

KNOWLEDGE_BASE_PATH = "./knowledge_base"
CHROMA_DB_PATH = "./chroma_db"

# --- NEW SECTION: Function to get data from Shortcut API ---
def get_shortcut_story_details(story_id: int) -> str:
    """Fetches a story from the Shortcut API and formats it for the LLM."""
    print(f"Fetching details for Shortcut story ID: {story_id}...")
    headers = {
        "Content-Type": "application/json",
        "Shortcut-Token": SHORTCUT_API_TOKEN,
    }
    try:
        response = requests.get(f"{SHORTCUT_BASE_URL}{story_id}", headers=headers)
        response.raise_for_status()  # This will raise an error for bad responses (4xx or 5xx)
        story_data = response.json()

        # Extract the title and the full description markdown
        story_title = story_data.get("name", "No Title Found")
        story_description = story_data.get("description", "No Description Found")
        
        # Format it nicely for the prompt, just like we did manually before
        formatted_text = f"**Feature:** {story_title}\n\n{story_description}"
        print("Successfully fetched and formatted story details.")
        return formatted_text

    except requests.exceptions.RequestException as e:
        print(f"Error fetching story from Shortcut API: {e}")
        return None
# --- END NEW SECTION ---


# --- Part 3: Knowledge Base Processing (Identical) ---
print("Step 1: Loading documents from the knowledge base...")
loader = DirectoryLoader(KNOWLEDGE_BASE_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)
print(f"Split documents into {len(docs)} chunks.")

print("Step 2: Creating local embeddings and storing them in ChromaDB...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory=CHROMA_DB_PATH
)
print("Knowledge base ready in ChromaDB.")


# --- Part 4: Define the AI Model and the Prompt (Identical) ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
prompt = ChatPromptTemplate.from_template("""
You are an expert Senior QA Engineer. Your task is to write detailed, comprehensive test cases based on the provided Acceptance Criteria and relevant context from our internal knowledge base.

Instructions:
1. Analyze the full story description, including its Context, Acceptance Criteria (AC), and Testing Approach.
2. Use the provided "Context" from the knowledge base documents to add specific, relevant details to your test cases.
3. Generate test cases in the Gherkin format (Given/When/Then).
4. Include Positive, Negative, and Edge Case scenarios.
5. If a testing approach is specified (e.g., test on specific products), create test suites for each item.

**Context from Knowledge Base:**
{context}

**Shortcut Story Details:**
{input}

**Your Generated Test Cases:**
""")

# --- Part 5: Create and Run the RAG Chain ---
print("Step 3: Setting up the RAG chain...")
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = vectorstore.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)
print("RAG chain is ready.")

# --- MODIFIED: Instead of a hardcoded string, we now call our new function ---
# Find the story number from your Shortcut URL. 
# For example, if the URL is ".../story/9432/...", the ID is 9432.
STORY_ID_TO_FETCH = 100390  # <--- CHANGE THIS TO YOUR STORY ID

story_content = get_shortcut_story_details(STORY_ID_TO_FETCH)

if story_content:
    print("\nStep 4: Generating test cases based on Shortcut story...")
    response = retrieval_chain.invoke({"input": story_content})

    # --- Part 6: Display the Result ---
    print("\n--- AI-Generated Test Cases (from Google Gemini) ---")
    print(response["answer"])
    print("------------------------------------------------------\n")
else:
    print("\nCould not generate test cases because the Shortcut story could not be fetched.")
# --- END MODIFIED ---
