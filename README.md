# Test-Case-Generator

🚀 AI-Powered Test Case Generator

## 1. Project Overview

### What is this tool?

This is an interactive web application that automates the creation of detailed, context-aware test cases. It takes a story ID from our project management tool (Shortcut) and relevant knowledge base documents (like PDFs of design specs) as input. It then generates a structured test plan in a clean, professional table format, which can be revised in a chat and exported to CSV.

### What problem does it solve?

Manually writing test cases is a slow, repetitive, and error-prone process. Testers often have to cross-reference multiple documents to ensure full coverage. This tool solves that problem by:

*   **Saving Time:** Drastically reducing the time it takes to write initial test drafts from hours to minutes.
*   **Improving Quality:** Ensuring test cases are consistent and directly linked to the latest documentation.
*   **Increasing Coverage:** The AI can identify complex process flows, edge cases, and negative scenarios that a human might overlook.
*   **Standardizing Output:** Generates a consistent, professional test plan format every time.

## 2. How it Works (The "Magic")

This is the most important part of the presentation. Explain it with this analogy:

> "Imagine you're a new QA engineer. To write good test cases for a new feature, you don't just look at the ticket. You also read the existing documentation—design docs, user guides, and information about related features.
>
> This tool does the exact same thing, but in seconds.
>
> It reads the knowledge base you give it (the PDFs).
> It reads the new ticket (from the Shortcut ID).
> It then uses its AI "brain" to connect the dots between the new requirements and the existing system knowledge, generating a comprehensive test plan that is far more detailed than if it only looked at the ticket alone."

This technique is called **Retrieval-Augmented Generation (RAG)**.

## 3. Key Features

*   **Dynamic Knowledge Base:** Upload any relevant PDF documents on-the-fly to provide context for the AI.
*   **Shortcut Integration:** Fetches story details directly using a story ID, ensuring the input is always accurate.
*   **S-Tier AI Model:** Powered by Google's Gemini 2.0 Flash, providing a state-of-the-art balance of intelligence, speed, and generous free-tier limits.
*   **Polished UI:** Displays the generated test cases in a clean, sortable, interactive table within the chat, not as raw text.
*   **Interactive Revisions:** The generated test cases are not final. You can chat with the AI to add, remove, or modify tests until they are perfect.
*   **Structured CSV Export:** The final test plan is exported as a clean CSV file, ready to be imported into Excel, Google Sheets, or a test management tool.

## 4. Technology Stack

*   **AI Model:** Google Gemini 2.0 Flash (via API)
*   **Web Framework:** Streamlit
*   **Core AI Library:** LangChain
*   **Document Processing:** PyPDFLoader (for PDFs), ChromaDB (for the vector store), SentenceTransformers (for embeddings)
*   **Data Handling & Export:** Pandas

## 5. Setup Instructions (For the Demo)

Follow these steps exactly to get the project running on any machine.

### Prerequisites

*   Python 3.8+ installed.
*   `git` installed (for cloning the repository).

### Step-by-Step Setup

1.  **Clone the Repository:**
    Open a terminal and run this command to download the project files:
    ```bash
    git clone <repository_url>
    cd <repository_folder_name>
    ```

2.  **Create a Virtual Environment:**
    This isolates the project's libraries. It's a crucial step.
    ```bash
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```
    Your terminal prompt should now start with `(venv)`.

3.  **Create the `requirements.txt` file:**
    Create a new file in the project folder named `requirements.txt` and paste the following lines into it. This file tells Python which libraries to install.
    ```
    streamlit
    requests
    langchain
    langchain-google-genai
    pypdf
    chromadb
    sentence-transformers
    pandas
    python-dotenv
    google-generativeai
    ```

4.  **Install the Libraries:**
    Run this command to install all the necessary packages from the file you just created:
    ```bash
    pip install -r requirements.txt
    ```

5.  **Set up the API Keys (Secrets):**
    The app needs two secret keys to work. We store them in a file that is kept private.
    
    Create a new file in the project folder named `.env`.
    Open the `.env` file and paste the following lines, replacing the placeholder text with the actual keys.
    ```dotenv
    # Get this from your Shortcut account (Settings > API Tokens)
    SHORTCUT_API_TOKEN="your-real-shortcut-token-goes-here"
    
    # Get this from Google AI Studio (https://aistudio.google.com/ > Get API key)
    GOOGLE_API_KEY="AIzaSy...your-real-google-api-key-goes-here"

    # OPTIONAL: If your organization uses a custom Shortcut URL, set it here.
    # If you leave this out, it will default to the standard "https://api.app.shortcut.com/api/v3/stories/"
    # SHORTCUT_API_BASE_URL="https://your-company.shortcut.com/api/v3/stories/"
    ```

## 6. How to Run the Application

Once the setup is complete, running the app is simple.

1.  Make sure you are in the project directory and your virtual environment `(venv)` is active.
2.  Run the following command in your terminal:
    ```bash
    streamlit run app.py
    ```
3.  A new tab should automatically open in your web browser with the application running. If not, the terminal will give you a local URL (like `http://localhost:8501`) to open manually.
