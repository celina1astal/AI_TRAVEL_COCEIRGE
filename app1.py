import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain_tavily import TavilySearch
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage, AIMessage

# --- 1. PAGE CONFIG & SECRETS ---
st.set_page_config(page_title="AI Travel Agent", page_icon="✈️", layout="wide")

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY")
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY or ""

# --- 2. RAG ENGINE (CACHED) ---
@st.cache_resource
def load_vector_db():
    if not os.path.exists("travel_sample.pdf"):
        st.error("Error: 'travel_sample.pdf' not found. Please ensure it is in your project root.")
        return None
    
    try:
        loader = PyPDFLoader("travel_sample.pdf")
        pages = loader.load()
        # Chunks optimized for Llama-3 reasoning
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
        docs = splitter.split_documents(pages)
        
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", 
            google_api_key=GEMINI_API_KEY
        )
        return FAISS.from_documents(docs, embeddings)
    except Exception as e:
        st.sidebar.error(f"PDF Vector Error: {e}")
        return None

vector_db = load_vector_db()

# --- 3. TOOL DEFINITIONS ---

@tool
def search_travel_pdf(query: str):
    """Searches the internal travel manual for specific itineraries, flight codes, or hotel lists."""
    if vector_db:
        docs = vector_db.similarity_search(query, k=3)
        return "\n".join([d.page_content for d in docs])
    return "Knowledge base not initialized. Check API keys."

web_search = TavilySearch(max_results=3, search_depth="advanced")
wiki_api = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1000)
wiki_search = WikipediaQueryRun(api_wrapper=wiki_api)

# Mapping dictionary to handle variant tool names returned by different LLMs
tools = [search_travel_pdf, web_search, wiki_search]
tool_map = {
    "search_travel_pdf": search_travel_pdf,
    "tavily_search_results_json": web_search,
    "tavily_search": web_search,
    "wikipedia": wiki_search
}

# --- 4. AGENT SETUP ---
llm = ChatGroq(
    model="llama-3.3-70b-versatile", 
    api_key=GROQ_API_KEY
).bind_tools(tools)

# --- 5. CHAT INTERFACE ---
st.title("✈️ AI Travel Concierge")

if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="""You are a professional travel agent. 
        - For greetings, reply politely without tools.
        - Use 'search_travel_pdf' for internal flight/plan data from the manual.
        - Use 'tavily_search_results_json' for live weather, prices, or current news.
        - Use 'wikipedia' for historical landmarks or general geography.""")
    ]

# Display history (Filtering out ToolMessages from UI for cleanliness)
for m in st.session_state.messages:
    if isinstance(m, HumanMessage):
        st.chat_message("user").write(m.content)
    elif isinstance(m, AIMessage) and m.content:
        st.chat_message("assistant").write(m.content)

# --- 6. AGENT EXECUTION LOOP ---
if user_input := st.chat_input("Ask me about your trip..."):
    st.chat_message("user").write(user_input)
    st.session_state.messages.append(HumanMessage(content=user_input))

    with st.chat_message("assistant"):
        # SAFETY: Trim history to last 6 messages + System Prompt to prevent Token Overflows (BadRequestError)
        # This keeps the context concise but relevant.
        history_buffer = [st.session_state.messages[0]] + st.session_state.messages[-6:]
        
        # First iteration
        response = llm.invoke(history_buffer)
        
        # Process Tool Calls (Limited to 3 iterations to prevent infinite loops)
        max_turns = 3
        while hasattr(response, 'tool_calls') and response.tool_calls and max_turns > 0:
            st.session_state.messages.append(response)
            
            for tool_call in response.tool_calls:
                t_name = tool_call["name"]
                t_args = tool_call["args"]
                
                with st.status(f"Acting as Concierge: {t_name}...", expanded=False) as status:
                    if t_name in tool_map:
                        result = tool_map[t_name].invoke(t_args)
                        status.update(label=f"Data retrieved via {t_name}", state="complete")
                    else:
                        result = f"Error: Tool '{t_name}' is not available."
                        status.update(label="Configuration Error", state="error")
                
                st.session_state.messages.append(
                    ToolMessage(content=str(result), tool_call_id=tool_call["id"])
                )
            
            # Re-summarize with new tool info
            history_buffer = [st.session_state.messages[0]] + st.session_state.messages[-10:]
            response = llm.invoke(history_buffer)
            max_turns -= 1

        # Final UI Output
        if response.content:
            st.markdown(response.content)
            st.session_state.messages.append(response)
