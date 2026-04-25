import streamlit as st
import requests

st.title("🧠 AI Copilot (Day 1)")

query = st.text_input("Ask something:")

if st.button("Submit"):
    response = requests.post(
        "http://127.0.0.1:8000/query",
        json={"query": query}
    )

    if response.status_code == 200:
        data = response.json()
        st.write("### Answer:")
        st.write(data["answer"])
        st.write(f"Score: {data['score']:.4f}")
    else:
        st.error("Error connecting to backend")