import streamlit as st
import pandas as pd
import io

# --- LangChain Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# --- Local Imports ---
from core_logic import (
    load_api_keys,
    get_shortcut_story_details,
    get_retriever_from_files,
    create_rag_chain,
    SHORTCUT_BASE_URL  # Import the default URL
)

# --- Load API Keys ---
try:
    load_api_keys()
except Exception as e:
    st.error(f"Failed to load API keys: {e}")

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

# --- Main App UI ---
st.set_page_config(page_title="AI Test Case Generator", layout="wide")
st.title("🚀 AI Test Case Generator")
st.subheader("Powered by Gemini 2.0 Flash")

if "retriever" not in st.session_state: st.session_state.retriever = None
if "messages" not in st.session_state: st.session_state.messages = []
if "latest_test_cases_md" not in st.session_state: st.session_state.latest_test_cases_md = ""

with st.sidebar:
    st.header("Setup")
    
    with st.expander("Advanced Settings"):
        shortcut_url_override = st.text_input(
            "Shortcut API Base URL", 
            value=SHORTCUT_BASE_URL, 
            help="The base URL for the Shortcut API, ending in `/stories/`. You can also set this with the `SHORTCUT_API_BASE_URL` environment variable."
        )

    # Updated file uploader to accept more types
    uploaded_files = st.file_uploader(
        "1. Upload Knowledge Base Documents", 
        type=["pdf", "docx", "doc", "txt", "md"], 
        accept_multiple_files=True
    )
    
    use_persistent_db = st.checkbox("Use Persistent Knowledge Base", value=False, help="If checked, uploaded documents will be added to a persistent database on disk. If unchecked, the knowledge base will be in-memory for this session only.")

    if uploaded_files:
        if st.button("Process Knowledge Base"):
            with st.spinner("Processing knowledge base..."):
                st.session_state.retriever = get_retriever_from_files(uploaded_files, use_persistent_db)
                st.success("Knowledge base is ready!")

    shortcut_id = st.text_input("2. Enter Shortcut Story ID", placeholder="e.g., 12345")

    if st.button("Generate Test Cases", type="primary"):
        if st.session_state.retriever and shortcut_id:
            with st.spinner("Generating with Gemini 2.0 Flash..."):
                story_content = get_shortcut_story_details(shortcut_id, base_url=shortcut_url_override)
                if story_content.startswith("Error:"):
                    st.error(story_content)
                else:
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

                    retrieval_chain = create_rag_chain(llm, st.session_state.retriever, initial_prompt)
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
