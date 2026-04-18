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

# --- 1. CONFIGURATION & ERROR HANDLING FOR SECRETS ---
st.set_page_config(page_title="✈️ AI Travel Concierge", layout="wide")
st.title("✈️ AI Travel Concierge")

try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]
except Exception as e:
    st.error("Missing API Keys! Please check your .streamlit/secrets.toml file.")
    st.stop()

# --- 2. DEFINE TOOLS WITH ERROR WRAPPERS ---

@tool
def search_travel_pdf(query: str):
    """Searches the local travel manual and flight itineraries for specific details."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
        # Ensure the 'faiss_index' folder exists from your ingestion script
        vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = vector_db.similarity_search(query, k=3)
        return "\n".join([d.page_content for d in docs])
    except Exception as e:
        return f"Error accessing PDF database: {str(e)}"

# Standard Tools
web_search = TavilySearchResults(tavily_api_key=TAVILY_API_KEY)
wiki_search = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

tools = [search_travel_pdf, web_search, wiki_search]
tool_map = {
    "search_travel_pdf": search_travel_pdf,
    "tavily_search_results_json": web_search,
    "wikipedia": wiki_search
}

# --- 3. INITIALIZE LLM WITH RETRIES ---
llm = ChatGroq(
    model="llama-3.1-8b-instant", # Using 8B for better stability/rate limits
    api_key=GROQ_API_KEY,
    max_retries=3
).bind_tools(tools)

# --- 4. SESSION STATE MANAGEMENT ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="You are a helpful Travel Concierge. Use tools to verify facts. If a tool fails, explain the issue to the user politely.")
    ]

# Display history
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage) and msg.content:
        st.chat_message("assistant").write(msg.content)

# --- 5. THE MAIN AGENT LOOP ---
if user_input := st.chat_input("Ask about your trip..."):
    st.session_state.messages.append(HumanMessage(content=user_input))
    st.chat_message("user").write(user_input)

    with st.chat_message("assistant"):
        try:
            # Step A: Initial LLM Call
            response = llm.invoke(st.session_state.messages)
            
            # Step B: Check for Tool Calls
            if response.tool_calls:
                st.session_state.messages.append(response)
                
                for tool_call in response.tool_calls:
                    t_name = tool_call["name"]
                    t_args = tool_call["args"]
                    
                    with st.status(f"Consulting {t_name}...", expanded=False) as status:
                        try:
                            # Execute tool safely
                            result = tool_map[t_name].invoke(t_args)
                            status.update(label=f"Finished {t_name}", state="complete")
                        except Exception as tool_err:
                            # Fallback error message sent back to LLM
                            result = f"Error: Tool '{t_name}' failed. {str(tool_err)}"
                            status.update(label=f"Error in {t_name}", state="error")
                        
                        st.session_state.messages.append(
                            ToolMessage(content=str(result), tool_call_id=tool_call["id"])
                        )
                
                # Step C: Final Response Generation
                final_response = llm.invoke(st.session_state.messages)
                st.write(final_response.content)
                st.session_state.messages.append(final_response)
            
            else:
                # Direct response without tools
                st.write(response.content)
                st.session_state.messages.append(response)

        except Exception as e:
            st.error("I'm having trouble connecting to my services. Please try again in a few seconds.")
            # Print for developer debugging in terminal
            print(f"CRITICAL ERROR: {e}")
