import streamlit as st
import requests

st.title("🧠 AI Copilot (RAG System)")

# Sidebar for document upload
with st.sidebar:
    st.header("Document Upload")
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
    
    if st.button("Process Document"):
        if uploaded_file is not None:
            with st.spinner("Processing and indexing document..."):
                # Send the file to backend
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                try:
                    response = requests.post("http://127.0.0.1:8000/upload", files=files)
                    if response.status_code == 200:
                        st.success(response.json()["message"])
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"Failed to connect to backend: {e}")
        else:
            st.warning("Please upload a PDF file first.")

# Main area for querying
st.header("Ask Questions")
query = st.text_input("Enter your query about the uploaded documents:")

if st.button("Submit"):
    if query:
        with st.spinner("Searching for answers..."):
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/query",
                    json={"query": query}
                )

                if response.status_code == 200:
                    data = response.json()
                    st.write("### Answer:")
                    st.info(data["answer"])
                    st.write(f"**Confidence Score:** {data['score']:.4f}")
                else:
                    st.error("Error retrieving response from backend")
            except Exception as e:
                st.error(f"Failed to connect to backend: {e}")
    else:
        st.warning("Please enter a query.")