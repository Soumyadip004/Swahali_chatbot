import os
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ========== Model & API config ==========
GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SYSTEM_PROMPT = (
    "You are a helpful AI assistant. "
    "Always follow the user's chosen language mode:\n"
    "- Auto: detect input (Swahili or English) and reply in the same language.\n"
    "- Swahili-only: reply only in Kiswahili, even if input is English.\n"
    "- English-only: reply only in English, even if input is Swahili."
)

def get_groq_client():
    if not GROQ_API_KEY:
        st.error("❌ GROQ_API_KEY not set in environment variables")
        st.stop()
    from groq import Groq
    return Groq(api_key=GROQ_API_KEY)

def generate_response(messages):
    """Generate response from Groq API"""
    try:
        client = get_groq_client()
        stream = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=300,
            stream=True,
        )
        
        response = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            response += delta
        
        return response
    
    except Exception as e:
        return f"Error: {str(e)}"

# ========== Streamlit App ==========
def main():
    # Page config
    st.set_page_config(
        page_title="🤖 Teknolojia - Bilingual AI Assistant",
        page_icon="🤖",
        layout="centered"
    )
    
    # Header
    st.title("🤖 **Teknolojia - Your bilingual AI assistant!**")
    st.markdown("🌐 Understands **English & Kiswahili**")
    st.markdown("⚡ Replies instantly with clarity and personality")
    st.markdown("💬 Built free & open for everyone to explore AI")
    
    # Language mode selection
    st.markdown("---")
    mode = st.selectbox(
        "🌍 Select Language Mode:",
        ["Auto", "Swahili-only", "English-only"],
        index=0,
        help="Auto: detects your language, Swahili-only: replies in Kiswahili, English-only: replies in English"
    )
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    st.markdown("---")
    st.markdown("### 💬 Chat")
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Type in English or Kiswahili…"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Build messages for API
                api_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                api_messages.extend(st.session_state.messages)
                api_messages.append({"role": "system", "content": f"User selected mode: {mode}"})
                
                # Generate response
                response = generate_response(api_messages)
                
                # Display response
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Sidebar with controls
    with st.sidebar:
        st.markdown("### 🔧 Controls")
        
        if st.button("🗑️ Clear Chat", type="secondary"):
            st.session_state.messages = []
            st.rerun()
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        This chatbot uses:
        - **Groq API** for fast inference
        - **Llama 3.1 8B** model
        - **Streamlit** for the interface
        
        Supports both English and Kiswahili!
        """)
        
        # Show message count
        msg_count = len([m for m in st.session_state.messages if m["role"] == "user"])
        st.metric("Messages sent", msg_count)

if __name__ == "__main__":
    main()
