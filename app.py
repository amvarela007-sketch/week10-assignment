import requests
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError
import datetime
import uuid
import json
import os

st.set_page_config(page_title="My AI Chat", layout="wide")

st.title("My AI Chat")

# Ensure chats directory exists
os.makedirs("chats", exist_ok=True)

try:
    TOKEN = st.secrets.get("HF_TOKEN", "")
except StreamlitSecretNotFoundError:
    st.error(
        "Secrets file not found or invalid. Please create `.streamlit/secrets.toml` with a valid `HF_TOKEN`."
    )
    st.stop()

if not TOKEN:
    st.error(
        "Hugging Face token not found. Please add `HF_TOKEN` to `.streamlit/secrets.toml` and restart the app."
    )
else:
    # Function to save chat to JSON file
    def save_chat(chat):
        filename = f"chats/{chat['id']}.json"
        chat_copy = chat.copy()
        chat_copy['timestamp'] = chat['timestamp'].isoformat()
        with open(filename, 'w') as f:
            json.dump(chat_copy, f, indent=4)

    # Function to load chat from JSON file
    def load_chat(chat_id):
        filename = f"chats/{chat_id}.json"
        with open(filename, 'r') as f:
            chat = json.load(f)
        chat['timestamp'] = datetime.datetime.fromisoformat(chat['timestamp'])
        return chat

    # Function to delete chat file
    def delete_chat_file(chat_id):
        filename = f"chats/{chat_id}.json"
        if os.path.exists(filename):
            os.remove(filename)

    # Function to load memory
    def load_memory():
        if os.path.exists("memory.json"):
            try:
                with open("memory.json", 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    # Function to save memory
    def save_memory(memory):
        with open("memory.json", 'w') as f:
            json.dump(memory, f, indent=4)

    # Function to query Hugging Face API (DEFINE BEFORE extract_memory)
    def query_hf(model_name: str, prompt_text: str, token: str, stream: bool = False, timeout: int = 30):
        # Use the Inference API with a full model path
        url = f"https://api-inference.huggingface.co/models/{model_name}"
        headers = {"Authorization": f"Bearer {token}"}
        payload = {"inputs": prompt_text}

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        except requests.exceptions.RequestException as exc:
            return False, f"Network error: {exc}"

        if resp.status_code == 401:
            return False, "Authentication failed. Check your Hugging Face token."
        if resp.status_code == 429:
            return False, "Rate limit exceeded. Please wait and try again."
        if resp.status_code == 404:
            return False, f"Model '{model_name}' not found. Try 'gpt2', 'distilgpt2', or another model."
        if resp.status_code >= 400:
            detail = resp.text.strip() or resp.reason
            return False, f"API error ({resp.status_code}): {detail}"

        try:
            data = resp.json()
        except ValueError:
            return False, "Invalid response from API."

        # Handle different response formats
        if isinstance(data, dict):
            if "error" in data:
                return False, f"API error: {data['error']}"
            if "generated_text" in data:
                return True, data["generated_text"]
            # Check for model load message
            if "estimated_time" in data:
                return False, f"Model is loading. Estimated time: {data['estimated_time']}s. Please try again."

        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict):
                if "generated_text" in first:
                    return True, first["generated_text"]
                if "error" in first:
                    return False, f"Error: {first['error']}"
            if isinstance(first, str):
                return True, first

        return True, str(data)

    # Function to extract memory from user message (DEFINED AFTER query_hf)
    def extract_memory(user_message, token):
        try:
            prompt = f"Extract personal facts from: '{user_message}'. Return JSON with keys: name, interests, preferences. If none found, return {{}}"
            success, result = query_hf("gpt2", prompt, token, stream=False)
            if success:
                try:
                    extracted = json.loads(result)
                    return extracted
                except:
                    return {}
            return {}
        except:
            return {}

    # Load memory
    memory = load_memory()

    # Initialize session state for chats
    if "chats" not in st.session_state:
        st.session_state.chats = []
        # Load existing chats from files
        if os.path.exists("chats"):
            for file in os.listdir("chats"):
                if file.endswith(".json"):
                    chat_id = file[:-5]
                    try:
                        chat = load_chat(chat_id)
                        st.session_state.chats.append(chat)
                    except Exception as e:
                        st.warning(f"Failed to load chat {chat_id}: {e}")

    if "current_chat_id" not in st.session_state:
        st.session_state.current_chat_id = None

    # Function to create a new chat
    def create_new_chat():
        chat_id = str(uuid.uuid4())
        new_chat = {
            "id": chat_id,
            "title": "New Chat",
            "timestamp": datetime.datetime.now(),
            "messages": []
        }
        st.session_state.chats.append(new_chat)
        st.session_state.current_chat_id = chat_id
        save_chat(new_chat)
        return chat_id

    # Function to delete a chat
    def delete_chat(chat_id):
        st.session_state.chats = [c for c in st.session_state.chats if c["id"] != chat_id]
        delete_chat_file(chat_id)
        if st.session_state.current_chat_id == chat_id:
            if st.session_state.chats:
                st.session_state.current_chat_id = st.session_state.chats[0]["id"]
            else:
                st.session_state.current_chat_id = None

    # Sidebar
    with st.sidebar:
        st.header("Chats")
        if st.button("New Chat", key="new_chat"):
            create_new_chat()
            st.rerun()

        st.markdown("---")

        # User Memory Section
        with st.expander("📝 User Memory"):
            st.write("**Stored traits:**")
            if memory:
                st.json(memory)
            else:
                st.info("No memory stored yet. Share preferences to build your profile!")
            
            if st.button("Clear Memory", key="clear_memory"):
                memory.clear()
                save_memory(memory)
                st.rerun()

        st.markdown("---")

        # Scrollable list of chats
        for chat in st.session_state.chats:
            col1, col2 = st.columns([4, 1])
            with col1:
                is_active = chat["id"] == st.session_state.current_chat_id
                button_label = f"{'▶ ' if is_active else ''}{chat['title']} - {chat['timestamp'].strftime('%H:%M')}"
                if st.button(
                    button_label,
                    key=f"chat_{chat['id']}",
                    help=f"Created: {chat['timestamp'].strftime('%Y-%m-%d %H:%M')}",
                    use_container_width=True
                ):
                    st.session_state.current_chat_id = chat["id"]
                    st.rerun()
            with col2:
                if st.button("✕", key=f"delete_{chat['id']}", help="Delete chat"):
                    delete_chat(chat["id"])
                    st.rerun()

    # Main area
    if st.session_state.current_chat_id:
        current_chat = next((c for c in st.session_state.chats if c["id"] == st.session_state.current_chat_id), None)
        if current_chat:
            st.subheader(f"Current Chat: {current_chat['title']}")
            # Display messages
            for msg in current_chat["messages"]:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
            # Input for new message
            if prompt := st.chat_input("Type your message..."):
                # Add user message to display
                current_chat["messages"].append({"role": "user", "content": prompt})
                save_chat(current_chat)
                st.rerun()  # Show user message immediately
        else:
            st.error("Selected chat not found.")
    else:
        st.info("No active chat. Create a new chat to start.")
    
    # Process AI response if last message is from user and we haven't responded yet
    if st.session_state.current_chat_id:
        current_chat = next((c for c in st.session_state.chats if c["id"] == st.session_state.current_chat_id), None)
        if current_chat and current_chat["messages"]:
            last_msg = current_chat["messages"][-1]
            if last_msg["role"] == "user":
                # Generate title if first message
                if len(current_chat["messages"]) == 1:
                    current_chat["title"] = last_msg["content"][:50] + ("..." if len(last_msg["content"]) > 50 else "")
                
                # Build conversation context with memory
                context = "You are a helpful AI assistant."
                if memory:
                    context += f" User info: {json.dumps(memory)}"
                
                conversation_text = context + "\n\n"
                for msg in current_chat["messages"]:
                    conversation_text += f"{msg['role'].capitalize()}: {msg['content']}\n"
                conversation_text += "Assistant:"
                
                # Get AI response
                with st.spinner("Thinking..."):
                    success, result = query_hf("gpt2", conversation_text, TOKEN, stream=False)
                
                if success:
                    # Save assistant response
                    current_chat["messages"].append({"role": "assistant", "content": result})
                    save_chat(current_chat)
                    
                    # Try to extract memory
                    extracted = extract_memory(last_msg["content"], TOKEN)
                    if extracted:
                        memory.update(extracted)
                        save_memory(memory)
                    
                    st.rerun()
                else:
                    st.error(f"Error: {result}")
