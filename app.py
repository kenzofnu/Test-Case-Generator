import streamlit as st
import os
import requests
from dotenv import load_dotenv
import pandas as pd
import io

# --- LangChain Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile

# --- Load API Keys ---
load_dotenv()
if "GOOGLE_API_KEY" not in os.environ or "SHORTCUT_API_TOKEN" not in os.environ:
    st.error("Please provide GOOGLE_API_KEY and SHORTCUT_API_TOKEN in your .env file.")
    st.stop()
    
SHORTCUT_API_TOKEN = os.environ["SHORTCUT_API_TOKEN"]
SHORTCUT_BASE_URL = "https://api.app.shortcut.com/api/v3/stories/"

# --- Helper Functions for Conversion ---

def convert_md_to_dataframe(markdown_string):
    """Converts a Markdown table string into a Pandas DataFrame, ignoring preceding text."""
    if not markdown_string:
        return None
    try:
        table_start_index = markdown_string.find('|')
        if table_start_index == -1: return None
        table_string = markdown_string[table_start_index:]
        lines = table_string.strip().split('\n')
        lines = [line for line in lines if not line.strip().startswith('|---')]
        string_io = io.StringIO('\n'.join(lines))
        df = pd.read_csv(string_io, sep='|', skipinitialspace=True)
        df = df.iloc[:, 1:-1]
        df.columns = df.columns.str.strip()
        df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        return df
    except Exception:
        return None

def convert_md_to_csv(markdown_string):
    """Converts a Markdown table string into a CSV string for download."""
    df = convert_md_to_dataframe(markdown_string)
    if df is not None:
        return df.to_csv(index=False)
    else:
        st.warning("Could not parse table for CSV export. Exporting raw text.")
        return markdown_string

# --- Core Functions (Identical) ---
@st.cache_resource
def get_embedding_function():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def process_uploaded_files(uploaded_files):
    docs = []
    with st.spinner("Processing knowledge base..."):
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            loader = PyPDFLoader(tmp_file_path)
            docs.extend(loader.load())
            os.remove(tmp_file_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    doc_splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=doc_splits, embedding=get_embedding_function())
    return vectorstore.as_retriever()

def get_shortcut_story_details(story_id):
    headers = {"Content-Type": "application/json", "Shortcut-Token": SHORTCUT_API_TOKEN}
    try:
        response = requests.get(f"{SHORTCUT_BASE_URL}{story_id}", headers=headers)
        response.raise_for_status()
        story_data = response.json()
        story_title = story_data.get("name", "No Title")
        story_description = story_data.get("description", "No Description")
        return f"**Feature:** {story_title}\n\n{story_description}"
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching story from Shortcut: {e}")
        return None

# --- Main App UI ---
st.set_page_config(page_title="AI Test Case Generator", layout="wide")
st.title("🚀 AI Test Case Generator")
st.subheader("Powered by Gemini 2.0 Flash")

if "retriever" not in st.session_state: st.session_state.retriever = None
if "messages" not in st.session_state: st.session_state.messages = []
if "latest_test_cases_md" not in st.session_state: st.session_state.latest_test_cases_md = ""

with st.sidebar:
    st.header("Setup")
    uploaded_files = st.file_uploader("1. Upload Knowledge Base PDFs", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        if st.button("Process Knowledge Base"):
            st.session_state.retriever = process_uploaded_files(uploaded_files)
            st.success("Knowledge base is ready!")
    shortcut_id = st.text_input("2. Enter Shortcut Story ID", placeholder="e.g., 12345")

    if st.button("Generate Test Cases", type="primary"):
        if st.session_state.retriever and shortcut_id:
            with st.spinner("Generating with Gemini 2.0 Flash..."):
                story_content = get_shortcut_story_details(shortcut_id)
                if story_content:
                    st.session_state.messages = []
                    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
                    
                    # --- THIS IS THE FULL, CORRECTED PROMPT ---
                    initial_prompt = ChatPromptTemplate.from_template("""
                    You are an expert Senior QA Engineer. Your task is to write detailed, comprehensive test cases in a structured table format.

                    **OUTPUT FORMAT INSTRUCTIONS:**
                    1.  You MUST output the test cases in a Markdown table.
                    2.  The table MUST have these exact seven columns: | Test Case | Test Description | Milestones | Preconditions | Test Step | Test Data | Expected Outcome |
                    3.  `Test Case`: A short, unique title for the test scenario.
                    4.  `Test Description`: A slightly more detailed sentence explaining the purpose of the test.
                    5.  `Milestones`: If any project milestones (like "Stage 2") are mentioned in the source documents, list them here. Otherwise, leave this blank.
                    6.  `Preconditions`: List the necessary state of the system before the test can begin (e.g., "User is logged in", "Previous order exists").
                    7.  `Test Step`: A numbered list of clear, concise actions the tester must perform. Use `<br>` for line breaks.
                    8.  `Test Data`: List any specific inputs, product names, or configurations needed for the test. Use `<br>` for line breaks.
                    9.  `Expected Outcome`: A clear description of the expected result after performing the steps.
                    10. Generate a comprehensive set of tests including positive, negative, and edge cases.

                    **CONTEXT FROM KNOWLEDGE BASE:**
                    {context}
                    
                    **SHORTCUT STORY DETAILS:**
                    {input}
                    
                    **YOUR GENERATED TEST CASES (MARKDOWN TABLE):**
                    """)

                    document_chain = create_stuff_documents_chain(llm, initial_prompt)
                    retrieval_chain = create_retrieval_chain(st.session_state.retriever, document_chain)
                    response = retrieval_chain.invoke({"input": story_content})
                    st.session_state.latest_test_cases_md = response["answer"]
                    st.session_state.messages.append(AIMessage(content=st.session_state.latest_test_cases_md))
                    st.success("Initial draft generated! You can now revise it below.")
        else:
            st.warning("Please process a knowledge base and enter a Story ID.")

# --- Main Chat Area ---
st.header("Generated Test Cases & Revisions")

if st.session_state.messages:
    for msg in st.session_state.messages:
        with st.chat_message("AI" if isinstance(msg, AIMessage) else "You"):
            df = convert_md_to_dataframe(msg.content)
            if df is not None:
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.write(msg.content)

    if revision_prompt := st.chat_input("Ask for revisions..."):
        st.session_state.messages.append(HumanMessage(content=revision_prompt))
        
        with st.spinner("Gemini 2.0 Flash is thinking..."):
            llm_revision = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

            # --- THIS IS THE FULL, CORRECTED REVISION PROMPT ---
            revision_prompt_template = ChatPromptTemplate.from_messages([
                ("system", """You are an expert QA assistant. Your goal is to revise the provided test cases, while strictly maintaining the required output format.

**RULES:**
1. The last AI message in the conversation contains the previous version of the test cases that you must edit.
2. You MUST ALWAYS respond with the FULL, updated set of test cases.
3. Your entire response MUST be in the specified Markdown table format: `| Test Case | Test Description | Milestones | Preconditions | Test Step | Test Data | Expected Outcome |`.
4. Do not add any conversational fluff. Only output the final Markdown table."""),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ])

            chain = revision_prompt_template | llm_revision
            response = chain.invoke({"chat_history": st.session_state.messages, "input": revision_prompt})
            
            st.session_state.latest_test_cases_md = response.content
            st.session_state.messages.append(AIMessage(content=st.session_state.latest_test_cases_md))
            st.rerun()
            
    st.download_button(
        label="📥 Export as CSV",
        data=convert_md_to_csv(st.session_state.latest_test_cases_md),
        file_name="generated_test_cases.csv",
        mime="text/csv",
    )
else:
    st.info("Set up your knowledge base and provide a Shortcut ID to begin.")
