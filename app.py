import requests
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

st.set_page_config(page_title="My AI Chat", layout="wide")

st.title("My AI Chat")

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
    model = "gpt2"
    prompt = "Hello!"

    def query_hf(model_name: str, prompt_text: str, token: str, timeout: int = 15):
        url = f"https://api-inference.huggingface.co/models/{model_name}"
        headers = {"Authorization": f"Bearer {token}"}
        payload = {"inputs": prompt_text}

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        except requests.exceptions.RequestException as exc:
            return False, f"Network error while contacting Hugging Face API: {exc}"

        if resp.status_code == 401:
            return False, "Authentication failed. Check that your Hugging Face token is valid."
        if resp.status_code == 429:
            return False, "Rate limit exceeded. Please wait and try again."
        if resp.status_code >= 400:
            detail = resp.text.strip() or resp.reason
            return False, f"Hugging Face API error ({resp.status_code}): {detail}"

        try:
            data = resp.json()
        except ValueError:
            return False, "Received an unexpected response from Hugging Face (not valid JSON)."

        if isinstance(data, dict) and "error" in data:
            return False, f"Hugging Face API error: {data['error']}"

        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict) and "generated_text" in first:
                return True, first["generated_text"]
            if isinstance(first, str):
                return True, first

        return True, str(data)

    with st.spinner("Sending test message to Hugging Face..."):
        success, result = query_hf(model, prompt, TOKEN)

    if success:
        st.subheader("Hugging Face response")
        st.write(result)
    else:
        st.error(result)
