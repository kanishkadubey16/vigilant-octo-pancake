import streamlit as st
import requests

# Page Configuration
st.set_page_config(page_title="AskVerse AI", page_icon="🌌", layout="wide")

# Custom CSS for Premium Dark Theme & ChatGPT-like UI
st.markdown("""
    <style>
    /* Main App Background */
    .stApp {
        background-color: #0E1117;
        color: #E0E0E0;
    }
    
    /* Title Styling */
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        background: -webkit-linear-gradient(#00FFA3, #03E1FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: #888;
        margin-top: -10px;
        margin-bottom: 2rem;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #161B22 !important;
        border-right: 1px solid #30363D;
    }

    /* Chat Input Styling */
    .stChatInputContainer {
        border-top: 1px solid #30363D !important;
        padding-top: 1rem;
    }

    /* Expander Styling */
    .stExpander {
        background-color: #161B22 !important;
        border: 1px solid #30363D !important;
        border-radius: 8px !important;
    }

    /* Button Styling */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        background-color: #21262D;
        color: #C9D1D9;
        border: 1px solid #30363D;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #30363D;
        border-color: #8B949E;
    }
    </style>
""", unsafe_allow_html=True)

# App Title & Subtitle
st.markdown('<h1 class="main-title">AskVerse AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Your intelligent document assistant powered by RAG</p>', unsafe_allow_html=True)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=80)
    st.header("Control Center")
    
    with st.expander("📂 Document Upload", expanded=True):
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")
        if st.button("🚀 Process & Index"):
            if uploaded_file is not None:
                with st.spinner("Analyzing document..."):
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    try:
                        response = requests.post("http://127.0.0.1:8000/upload", files=files)
                        if response.status_code == 200:
                            st.success("Document Indexed!")
                        else:
                            st.error("Upload failed.")
                    except Exception as e:
                        st.error(f"Backend Error: {e}")
            else:
                st.warning("Please select a file.")

    st.divider()
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📄 Source References"):
                for i, source in enumerate(message["sources"]):
                    st.markdown(f"**Chunk {i+1}**")
                    st.caption(source)

# User Interaction
if prompt := st.chat_input("Ask Verse anything..."):
    # Append User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate AI Response
    with st.chat_message("assistant"):
        with st.spinner("Synthesizing..."):
            try:
                # Prepare history (last 5 rounds)
                history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
                
                response = requests.post(
                    "http://127.0.0.1:8000/query",
                    json={"query": prompt, "history": history}
                )

                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    sources = data.get("sources", [])
                    
                    st.markdown(answer)
                    if sources:
                        with st.expander("📄 Source References"):
                            for i, source in enumerate(sources):
                                st.markdown(f"**Chunk {i+1}**")
                                st.caption(source)
                    
                    # Store Assistant Message
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "sources": sources
                    })
                else:
                    st.error("The AI is currently unavailable.")
            except Exception as e:
                st.error(f"Connection lost: {e}")