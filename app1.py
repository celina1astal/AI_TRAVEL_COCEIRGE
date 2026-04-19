import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="✈️ AI Travel Concierge", layout="wide")

# --- 2. [FIX] INITIALIZE SESSION STATE FIRST ---
# This MUST come before the sidebar to prevent "AttributeError"
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="You are a professional Travel Concierge. Use tools to verify details. Always cite which [Document Source] you are using if information comes from the PDF tool.")
    ]

# --- 3. SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Agent Control Panel")
    
    # Selection for Gemini Models (Free Tier compliant)
    model_choice = st.selectbox("🧠 Select Brain", ["gemini-1.5-flash", "gemini-1.5-pro"])
    temp = st.slider("🔥 Temperature", 0.0, 1.0, 0.4)
    
    st.divider()
    
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

    if st.button("🗑️ Reset Chat"):
        st.session_state.messages = [st.session_state.messages[0]]
        st.rerun()

# --- 4. API KEY VALIDATION ---
try:
    # We only need Gemini and Tavily now
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]
except Exception:
    st.error("Missing API Keys! Check .streamlit/secrets.toml")
    st.stop()

# --- 5. TOOLS ---
@tool
def search_travel_pdf(query: str):
    """Searches the local travel manual and flight itineraries for specific details."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
        vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = vector_db.similarity_search(query, k=3)
        
        context = "Information found in your local documents:\n"
        for i, d in enumerate(docs):
            context += f"\n[Document Source {i+1}]: {d.page_content}\n"
        return context
    except Exception as e:
        return f"Error accessing PDF database: {str(e)}"

web_search = TavilySearchResults(tavily_api_key=TAVILY_API_KEY)
wiki_search = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

tools = [search_travel_pdf, web_search, wiki_search]
tool_map = {
    "search_travel_pdf": search_travel_pdf,
    "tavily_search_results_json": web_search,
    "wikipedia": wiki_search
}

# --- 6. INITIALIZE GEMINI LLM ---
llm = ChatGoogleGenerativeAI(
    model=model_choice,
    google_api_key=GEMINI_API_KEY,
    temperature=temp
).bind_tools(tools)

# --- 7. UI AND CHAT LOOP ---
st.title("✈️ AI Travel Concierge")

for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage) and msg.content:
        st.chat_message("assistant").write(msg.content)

if user_input := st.chat_input("Ask about your trip..."):
    st.session_state.messages.append(HumanMessage(content=user_input))
    st.chat_message("user").write(user_input)

    with st.chat_message("assistant"):
        try:
            # Agent logic
            response = llm.invoke(st.session_state.messages)
            
            if response.tool_calls:
                st.session_state.messages.append(response)
                
                for tool_call in response.tool_calls:
                    t_name = tool_call["name"]
                    t_args = tool_call["args"]
                    
                    with st.status(f"Consulting {t_name}...", expanded=False) as status:
                        try:
                            # Use tool_map to call the correct function
                            result = tool_map[t_name].invoke(t_args)
                            status.update(label=f"Completed {t_name}", state="complete")
                        except Exception as tool_err:
                            result = f"Error in {t_name}: {str(tool_err)}"
                            status.update(label=f"Failed {t_name}", state="error")
                        
                        st.session_state.messages.append(
                            ToolMessage(content=str(result), tool_call_id=tool_call["id"])
                        )
                
                final_response = llm.invoke(st.session_state.messages)
                st.write(final_response.content)
                st.session_state.messages.append(final_response)
            
            else:
                st.write(response.content)
                st.session_state.messages.append(response)

        except Exception as e:
            st.error("Gemini is currently busy or Rate Limited. Please wait a moment.")
            print(f"Error: {e}")
