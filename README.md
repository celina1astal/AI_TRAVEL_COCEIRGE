
 ## **1.✈️ AI Travel Concierge: Multi-Modal RAG Agent**

*A production-ready AI agent that combines private document retrieval (RAG), live web search, and encyclopedic knowledge to provide hyper-local travel assistance.*

-----

## **2. 🛠️ Tech Stack**

  * **LLM Engine:** Llama 3.3 (via Groq Cloud)
  * **Orchestration:** LangChain (Agentic Workflow)
  * **Vector Database:** FAISS
  * **Embeddings:** Google Gemini (`gemini-embedding-001`)
  * **Frontend:** Streamlit
  * **Live Data APIs:** Tavily (Web Search), Wikipedia API

-----

## **3. 🌟 Key Features**

  * **Contextual RAG:** Answers questions based on a private travel manual (PDF) using semantic search.
  * **Real-time Awareness:** Fetches live weather, flight delays, and local news via Tavily.
  * **Encyclopedic Context:** Automatically pulls historical landmark data from Wikipedia.
  * **Memory Management:** Implements a sliding window buffer to maintain conversation context within API token limits.

-----

**4. 🚀 Quick Start**

**Prerequisites**

  * Python 3.10+
  * API Keys for **Groq**, **Google AI (Gemini)**, and **Tavily**.

**Installation**

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/ai-travel-concierge.git
    cd ai-travel-concierge
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Setup Secrets:**
    Create a `.streamlit/secrets.toml` file and add:
    ```toml
    GROQ_API_KEY = "your_key_here"
    GEMINI_API_KEY = "your_key_here"
    TAVILY_API_KEY = "your_key_here"
    ```
4.  **Run the app:**
    ```bash
    streamlit run app1.py
    ```

-----

## **5. 🏗️ System Architecture**

Explain the **"Brain"** of our app :

1.  **User Input:** Captured via Streamlit.
2.  **The Router:** LangChain analyzes the intent and decides whether to use a tool or answer directly.
3.  **Tool Execution:** \* **PDF:** Text chunks are vectorized and retrieved from FAISS.
      * **Web/Wiki:** APIs are called for external knowledge.
4.  **Final Synthesis:** The LLM combines the tool results into a polite, human-like travel advice.

-----

## **6. 📝 Sample Queries to Test**

  * *"What are key topics mentioned in pdf?"* (Triggers **PDF**)
  * *"What is the temperature in Bengaluru right now?"* (Triggers **Tavily**)
  * *"When to travel to goa?"* (Triggers **Wikipedia**)



