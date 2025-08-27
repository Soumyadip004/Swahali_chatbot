import os
import gradio as gr
import time
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
        raise RuntimeError("❌ GROQ_API_KEY not set")
    from groq import Groq
    return Groq(api_key=GROQ_API_KEY)

# ========== Chat logic ==========
def respond(message, chat_history, mode_value):
    """Main function that handles the chat interaction"""
    client = get_groq_client()
    
    # Build messages from chat history
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Add chat history
    for user_msg, assistant_msg in chat_history:
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if assistant_msg:
            messages.append({"role": "assistant", "content": assistant_msg})
    
    # Add mode and current message
    messages.append({"role": "system", "content": f"User selected mode: {mode_value}"})
    messages.append({"role": "user", "content": message})
    
    try:
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
        
        # Add to chat history
        chat_history.append((message, response))
        
        # Return empty string for textbox and updated chat history
        return "", chat_history
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        chat_history.append((message, error_msg))
        return "", chat_history

# ========== UI ==========
with gr.Blocks(title="🔄 DualBot") as demo:
    gr.Markdown("# 🤖**Teknolojia - Your bilingual AI assistant!**")
    gr.Markdown("🌐Understands **English & Kiswahili**")
    gr.Markdown("⚡Replies instantly with clarity and personality")
    gr.Markdown("💬Built free & open for everyone to explore AI")
    gr.Markdown("Select a mode and start chatting:")
    
    mode = gr.Radio(
        ["Auto", "Swahili-only", "English-only"],
        value="Auto",
        label="Language Mode",
    )
    
    # Fix the chatbot type warning
    chatbot = gr.Chatbot(height=400, type="tuples")  # Explicitly set type
    msg = gr.Textbox(placeholder="Type in English or Kiswahili…", lines=3)
    send = gr.Button("⌯⌲ Send")
    clear = gr.Button("🗑️ Clear Chat")
    
    # Connect the send button - returns 2 values: empty string and updated chatbot
    send.click(
        fn=respond,
        inputs=[msg, chatbot, mode],
        outputs=[msg, chatbot],  # Clear textbox and update chatbot
    )
    
    # Connect enter key
    msg.submit(respond, [msg, chatbot, mode], [msg, chatbot])
    
    # Clear chat function
    clear.click(lambda: [], None, chatbot, queue=False)

if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=8080, show_api=False)