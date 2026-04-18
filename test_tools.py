import os
from app1 import search_travel_pdf, web_search, wiki_search

def test_system():
    print("--- 🔍 STARTING SYSTEM DIAGNOSTICS ---")
    
    # 1. Test PDF Retrieval (RAG)
    print("\n[1/3] Testing PDF Tool...")
    try:
        # Using a keyword likely in your travel/lake reports
        pdf_res = search_travel_pdf.invoke({"query": "Bengaluru"})
        if pdf_res and "Error" not in pdf_res:
            print("✅ PDF Tool: SUCCESS (Data retrieved)")
        else:
            print("⚠️ PDF Tool: FAILED (No data found or search error)")
    except Exception as e:
        print(f"❌ PDF Tool: CRASHED ({e})")

    # 2. Test Web Search (Tavily)
    print("\n[2/3] Testing Tavily Web Search...")
    try:
        web_res = web_search.invoke({"query": "Current weather in Bengaluru"})
        if "temp" in str(web_res).lower() or "weather" in str(web_res).lower():
            print("✅ Web Tool: SUCCESS (Live data received)")
        else:
            print("⚠️ Web Tool: WARNING (Received response, but no weather data)")
    except Exception as e:
        print(f"❌ Web Tool: CRASHED (Check your Tavily API Key)")

    # 3. Test Wikipedia
    print("\n[3/3] Testing Wikipedia Tool...")
    try:
        wiki_res = wiki_search.invoke({"query": "VTU University"})
        if wiki_res and len(wiki_res) > 50:
            print("✅ Wikipedia: SUCCESS (History retrieved)")
        else:
            print("⚠️ Wikipedia: FAILED (Short or empty response)")
    except Exception as e:
        print(f"❌ Wikipedia: CRASHED ({e})")

    print("\n--- 🏁 DIAGNOSTICS COMPLETE ---")

if __name__ == "__main__":
    test_system()
