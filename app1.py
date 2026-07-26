import streamlit as st
import os
import sqlite3
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

# --- DATABASE SETUP ---
def init_db():
    conn = sqlite3.connect('travel_data.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS travel_history 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  user_query TEXT, 
                  ai_response TEXT, 
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

def save_to_db(query, response):
    conn = sqlite3.connect('travel_data.db')
    c = conn.cursor()
    c.execute("INSERT INTO travel_history (user_query, ai_response) VALUES (?, ?)", (query, response))
    conn.commit()
    conn.close()

def init_itinerary_db():
    conn = sqlite3.connect('travel_data.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS saved_itineraries 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  destination TEXT, 
                  duration INTEGER, 
                  content TEXT, 
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

# Initialize databases
init_db()
init_itinerary_db()


# --- 1. CONFIGURATION ---
st.set_page_config(page_title="✈️ AI Travel Concierge", layout="wide")


# --- 2. API KEY VALIDATION ---
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]
except Exception:
    st.error("Missing API Keys! Check .streamlit/secrets.toml")
    st.stop()


# --- 3. INITIALIZE BASE TOOLS ---
web_search = TavilySearchResults(tavily_api_key=TAVILY_API_KEY)
wiki_search = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())


# --- 4. CUSTOM TOOLS ---
@tool
def fetch_travel_deals(query: str):
    """
    Specialized tool to fetch real-time flight prices, hotel availability, 
    and travel itineraries using the Tavily travel-search optimization.
    """
    travel_query = f"{query} live prices flights hotels booking.com tripadvisor"
    return web_search.invoke({"query": travel_query})

@tool
def search_travel_pdf(query: str):
    """Searches the local travel manual and flight itineraries for specific details."""
    try:
        import os
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        index_dir = "faiss_index"
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
        
        # Check if the FAISS index folder exists
        if not os.path.exists(index_dir):
            doc_dir = "uploaded_documents"
            # If no files have been uploaded yet, return an instruction message
            if not os.path.exists(doc_dir) or not os.listdir(doc_dir):
                return "Error: No travel documents have been indexed yet. Please upload a PDF manual in the sidebar first."
            
            # Combine all uploaded PDFs into a single searchable index
            all_docs = []
            for file in os.listdir(doc_dir):
                if file.endswith(".pdf"):
                    loader = PyPDFLoader(os.path.join(doc_dir, file))
                    all_docs.extend(loader.load())
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            split_docs = text_splitter.split_documents(all_docs)
            
            vector_db = FAISS.from_documents(split_docs, embeddings)
            vector_db.save_local(index_dir)
        else:
            vector_db = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
            
        docs = vector_db.similarity_search(query, k=3)
        context = "Information found in your local documents:\n"
        for i, d in enumerate(docs):
            page_num = d.metadata.get('page', 0) + 1
            context += f"\n[Document Source {i+1} (Page {page_num})]: {d.page_content}\n"
        return context
        
    except Exception as e:
        return f"Error accessing PDF database: {str(e)}"


tools = [fetch_travel_deals, search_travel_pdf, web_search, wiki_search]
tool_map = {
    "fetch_travel_deals": fetch_travel_deals,
    "search_travel_pdf": search_travel_pdf,
    "tavily_search_results_json": web_search,
    "wikipedia": wiki_search
}


# --- 5. SESSION STATE INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="You are an expert Travel Concierge. "
            "IMPORTANT: Always use the fetch_travel_deals tool to get real-time flight and hotel prices. "
            "Structure your response with: 1. Day-by-Day Breakdown, 2. Interactive Map Suggestions, "
            "and 3. A final 'Budget Summary' table.")
    ]
if "run_agent" not in st.session_state:
    st.session_state.run_agent = False
if "last_query" not in st.session_state:
    st.session_state.last_query = ""


# --- 6. QUICK ITINERARY GENERATOR ---
with st.expander("📅 Quick Itinerary Generator", expanded=True):
    with st.form("itinerary_form"):
        dest = st.text_input("Where do you want to go?")
        days = st.number_input("Number of Days", min_value=1, max_value=14, value=3)
        budget = st.selectbox("Budget Level", ["Economy", "Standard", "Luxury"])
        submitted = st.form_submit_button("Generate Trip Plan")

        if submitted and dest:
            itinerary_prompt = f"Plan a {days}-day {budget} trip to {dest}. Use your fetch_travel_deals tool for real prices."
            st.session_state.messages.append(HumanMessage(content=itinerary_prompt))
            st.session_state.last_query = itinerary_prompt
            st.session_state.run_agent = True
            st.rerun()


# --- 7. SIDEBAR & THEME LOGIC ---
with st.sidebar:
    st.title("⚙️ Settings")
    temp = st.slider("Temperature", 0.0, 1.0, 0.4)
    theme_choice = st.selectbox(
        "Select UI Theme", 
        ["Corporate Blue", "Nature Green", "Deep Sea", "Sunset Orange"]
    )

    theme_colors = {
        "Deep Sea": {"primary": "#007BFF", "hover": "#0056b3"},
        "Nature Green": {"primary": "#28a745", "hover": "#218838"},
        "Corporate Blue": {"primary": "#17a2b8", "hover": "#117a8b"},
        "Sunset Orange": {"primary": "#fd7e14", "hover": "#d35400"}
    }

    selected_color = theme_colors[theme_choice]["primary"]
    hover_color = theme_colors[theme_choice]["hover"]

    st.divider()

    st.subheader("📁 Knowledge Base Base")
    uploaded_file = st.file_uploader("Upload Travel Manual (PDF)", type=["pdf"])
    
    if uploaded_file is not None:
        import shutil
        # Create directories safely if they are missing
        os.makedirs("uploaded_documents", exist_ok=True)
        file_path = os.path.join("uploaded_documents", uploaded_file.name)
        
        # Save the uploaded file to disk
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    # Re-build the FAISS index automatically to include the new document
        with st.spinner("Processing PDF and indexing text vectors..."):
            try:
                from langchain_community.document_loaders import PyPDFLoader
                from langchain_text_splitters import RecursiveCharacterTextSplitter
                
                # Force delete old index if it exists so it updates cleanly
                if os.path.exists("faiss_index"):
                    shutil.rmtree("faiss_index")
                    
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                split_docs = text_splitter.split_documents(docs)
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
                vector_db = FAISS.from_documents(split_docs, embeddings)
                vector_db.save_local("faiss_index")
                
                st.sidebar.success(f"✅ Loaded: {uploaded_file.name}")
            except Exception as index_err:
                st.sidebar.error(f"Failed to process text: {str(index_err)}")
    
    st.divider()
    
    st.subheader("📜 Recent Travels")

    if st.button("🗑️ Clear History"):
        conn = sqlite3.connect('travel_data.db')
        c = conn.cursor()
        c.execute("DELETE FROM travel_history")
        conn.commit()
        conn.close()
        st.success("History Cleared!")
        st.rerun()
        
    try:
        conn = sqlite3.connect('travel_data.db')
        history = conn.execute("SELECT DISTINCT user_query FROM travel_history ORDER BY id DESC LIMIT 5").fetchall()
        conn.close()
        
        if history:
            for item in history:
                st.caption(f"📍 {item[0]}")
        else:
            st.write("No recent searches yet.")
    except Exception as e:
        st.caption("History currently unavailable.")

    def export_chat():
        chat_str = "AI TRAVEL CONCIERGE LOG\n" + "="*30 + "\n"
        for msg in st.session_state.messages:
            if not isinstance(msg, SystemMessage):
                role = "User" if isinstance(msg, HumanMessage) else "Assistant"
                content = msg.content if msg.content else "[Tool Invocations]"
                chat_str += f"{role}: {content}\n\n"
        return chat_str

    st.download_button(
        label="⬇️ Download Chat Log",
        data=export_chat(),
        file_name="travel_agent_log.txt",
        mime="text/plain"
    )
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = [st.session_state.messages[0]]
        st.session_state.run_agent = False
        st.session_state.last_query = ""
        st.rerun()


# --- 8. DYNAMIC CSS ---
st.markdown(f"""
    <style>
    div.stButton > button {{
        background-color: {selected_color} !important;
        color: white !important;
        border-radius: 8px !important;
        transition: 0.3s !important;
        width: 100% !important;
    }}
    div.stButton > button:hover {{
        background-color: {hover_color} !important;
        border: 1px solid white !important;
    }}
    .stDownloadButton > button {{
        background-color: {selected_color} !important;
        color: white !important;
        width: 100% !important;
    }}
    </style>
""", unsafe_allow_html=True)


# --- 9. INITIALIZE LLM ---
llm = ChatGroq(
    model="llama-3.3-70b-versatile", 
    api_key=GROQ_API_KEY,
    temperature=temp,
    max_retries=3
).bind_tools(tools)


# --- 10. UI DISPLAY & CHAT HISTORY ---
st.title("✈️ AI Travel Concierge")

for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user", avatar="😎"):
            st.markdown(msg.content)
    elif isinstance(msg, (ToolMessage, SystemMessage)):
        continue
    else:
        if msg.content:
            with st.chat_message("assistant", avatar="👾"):
                st.markdown(msg.content)


# --- 11. CHAT INPUT DETECTION ---
user_input = st.chat_input("Ask about your trip...")

if user_input:
    st.session_state.messages.append(HumanMessage(content=user_input))
    st.session_state.last_query = user_input
    st.session_state.run_agent = True
    st.rerun() 


# --- 12. RECURSIVE AGENTIC PROCESSING LOOP ---# 
if st.session_state.run_agent:
    st.session_state.run_agent = False # Instantly clear state flag
    
    with st.chat_message("assistant", avatar="🤖"):
        try:
            max_iterations = 5
            iterations = 0
            
            response = llm.invoke(st.session_state.messages)
            
            while response.tool_calls and iterations < max_iterations:
                st.session_state.messages.append(response)
                iterations += 1
                
                for tool_call in response.tool_calls:
                    t_name = tool_call["name"]
                    t_args = tool_call["args"]

                    with st.status(f"Acting as Agent: {t_name}...", expanded=False) as status:
                        try:
                            result = tool_map[t_name].invoke(t_args)
                            status.update(label=f"Completed {t_name}", state="complete")
                        except Exception as tool_err:
                            result = f"Technical Failure in {t_name}: {str(tool_err)}"
                            status.update(label=f"Error in {t_name}", state="error")

                        st.session_state.messages.append(
                            ToolMessage(content=str(result), tool_call_id=tool_call["id"])
                        )
                
                response = llm.invoke(st.session_state.messages)

            # Display final text answer to user
            st.markdown(response.content)
            st.session_state.messages.append(response)
            
            # Save final response context to SQLite DB
            save_to_db(st.session_state.last_query, response.content)
            st.rerun()

        except Exception as e:
            st.error("I encountered an issue processing your request.")
            st.caption(f"DEV LOG: {str(e)}")



