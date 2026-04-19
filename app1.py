import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

# --- 1. CONFIGURATION & SIDEBAR ---
st.set_page_config(page_title="✈️ AI Travel Concierge", layout="wide")

with st.sidebar:
    st.title("🛠️ Agent Control Panel")
    st.info("B.E. Computer Science Project - RVITM")
    
    # Model Selection
    model_choice = st.selectbox("LLM Model", ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"])
    temp = st.slider("Temperature (Creativity)", 0.0, 1.0, 0.4)
    
    st.divider()
    
    # Chat Export Feature
    def export_chat():
        chat_str = "AI TRAVEL CONCIERGE LOG\n" + "="*30 + "\n"
        for msg in st.session_state.messages:
            if not isinstance(msg, SystemMessage):
                role = "User" if isinstance(msg, HumanMessage) else "Assistant"
                chat_str += f"{role}: {msg.content}\n\n"
        return chat_str

    st.download_button(
        label="📥 Download Chat Log",
        data=export_chat(),
        file_name="travel_agent_log.txt",
        mime="text/plain"
    )
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = [st.session_state.messages[0]]
        st.rerun()

# --- 2. API KEY VALIDATION ---
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]
except Exception:
    st.error("Missing API Keys! Check .streamlit/secrets.toml")
    st.stop()

# --- 3. TOOLS WITH SOURCE TAGGING ---

@tool
def search_travel_pdf(query: str):
    """Searches the local travel manual and flight itineraries for specific details."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
        # allow_dangerous_deserialization is required for loading local FAISS files
        vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = vector_db.similarity_search(query, k=3)
        
        # Adding Source Tagging for 'Explainability'
        context = "Information found in your local documents:\n"
        for i, d in enumerate(docs):
            context += f"\n[Document Source {i+1}]: {d.page_content}\n"
        return context
    except Exception as e:
        return f"Error accessing PDF database: {str(e)}"

# External Search Tools
web_search = TavilySearchResults(tavily_api_key=TAVILY_API_KEY)
wiki_search = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

tools = [search_travel_pdf, web_search, wiki_search]
tool_map = {
    "search_travel_pdf": search_travel_pdf,
    "tavily_search_results_json": web_search,
    "wikipedia": wiki_search
}

# --- 4. INITIALIZE LLM ---
llm = ChatGroq(
    model=model_choice, 
    api_key=GROQ_API_KEY,
    temperature=temp,
    max_retries=3
).bind_tools(tools)

# --- 5. SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="You are a professional Travel Concierge. Use tools to verify details. Always cite which [Document Source] you are using if information comes from the PDF tool.")
    ]

# UI Title
st.title("✈️ AI Travel Concierge")

# Display Messages
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage) and msg.content:
        st.chat_message("assistant").write(msg.content)

# --- 6. AGENTIC LOOP WITH ERROR HANDLING ---
if user_input := st.chat_input("Ask about your trip..."):
    st.session_state.messages.append(HumanMessage(content=user_input))
    st.chat_message("user").write(user_input)

    with st.chat_message("assistant"):
        try:
            # First pass: Determine if tools are needed
            response = llm.invoke(st.session_state.messages)
            
            if response.tool_calls:
                st.session_state.messages.append(response)
                
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
                
                # Final synthesis pass
                final_response = llm.invoke(st.session_state.messages)
                st.write(final_response.content)
                st.session_state.messages.append(final_response)
            
            else:
                st.write(response.content)
                st.session_state.messages.append(response)

        except Exception as e:
            st.error("I encountered a connection error. This is usually due to API Rate Limits.")
            st.warning("Try switching to the '8B-Instant' model in the sidebar for faster responses.")
            print(f"DEV LOG: {e}")
