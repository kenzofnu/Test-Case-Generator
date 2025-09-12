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
    ```

## 6. How to Run the Application

Once the setup is complete, running the app is simple.

1.  Make sure you are in the project directory and your virtual environment `(venv)` is active.
2.  Run the following command in your terminal:
    ```bash
    streamlit run app.py
    ```
3.  A new tab should automatically open in your web browser with the application running. If not, the terminal will give you a local URL (like `http://localhost:8501`) to open manually.

## 7. The Demo Flow (Presentation Script)

Here is a script for presenting the application.

*   **Introduction:** Start with the "Problem Statement" and the "Core Concept" analogy explained above.
*   **Step 1: Upload Knowledge Base:**
    On the left sidebar, use the file uploader to select one or more PDF documents. These act as the AI's "long-term memory."
    Explain that these could be technical specifications, design documents, or process flows.
*   **Step 2: Process the Knowledge:**
    Click the "Process Knowledge Base" button.
    Explain that the app is now reading the PDFs and creating a searchable index so the AI can find relevant information instantly.
*   **Step 3: Enter the Story ID:**
    Enter a valid numeric ID from Shortcut into the text box.
*   **Step 4: Generate the First Draft:**
    Click the "Generate Test Cases" button.
    Point out how the AI generates a beautiful, structured table, not just raw text. This is a key feature.
    Explain that the app is fetching the story, searching the knowledge base, and sending all of that information to the Gemini AI to generate the initial test plan.
*   **Step 5: Interactive Revisions:**
    Use the chat box at the bottom to ask for a change. Be specific.
    
    > Good demo commands:
    > *   "Add a new negative test case for an incorrect user role."
    > *   "In the first test case, change the 'Expected Outcome' to also include a success notification."
    > *   "Can you combine the two tests related to invalid PIIDs into a single scenario?"
    
    After each request, point out how the AI regenerates the entire table with the requested changes, maintaining the professional format.
*   **Step 6: Export the Final Result:**
    Once you are happy with the test plan, click the "Export as CSV" button.
    Open the downloaded `.csv` file in Excel or Google Sheets to show that it is a clean, structured file ready for use in other tools.

## 8. Troubleshooting

If something goes wrong during the demo, here are the most likely causes:

*   **Error:** `ModuleNotFoundError`: The `pip install -r requirements.txt` command was not run, or you are not in the `venv`.
*   **Error:** `API Key not found`: The `.env` file is missing, named incorrectly, or doesn't contain the correct keys.
*   **Error:** `429 Resource Exhausted (Rate Limit)`: You are making revisions too quickly. The Gemini 2.0 Flash model has a high limit (15 requests per minute), but it's not infinite. Wait about 5-10 seconds between revision requests. If this happens, just say "Looks like we're a bit too fast for the free tier, let's give it a moment" and wait 20 seconds before trying again.