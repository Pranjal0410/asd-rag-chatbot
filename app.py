import streamlit as st
import os
from chain import ask, build_prompt
from retriever import retrieve
from groq import Groq
from ingest import index_resume

if not os.path.exists("./chroma_db"):
    index_resume()

st.set_page_config(page_title="ASD Notes Chatbot", page_icon="📚")
st.title("📚 Advanced System Design — Notes Chatbot")
st.caption("Sirf PDF se answer dega — page number ke saath")

# API key input
groq_key = st.text_input("Groq API Key", type="password")
if groq_key:
    os.environ["GROQ_API_KEY"] = groq_key

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if groq_key:
    if question := st.chat_input("Kuch bhi poochho ASD ke baare mein..."):
        with st.chat_message("user"):
            st.write(question)
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("assistant"):
            with st.spinner("Soch raha hoon..."):
                chunks = retrieve(question)
                prompt = build_prompt(question, chunks)

                client = Groq()
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}]
                )
                answer = response.choices[0].message.content

            st.write(answer)

            if chunks:
                with st.expander("📄 Sources dekho"):
                    for chunk in chunks:
                        st.markdown(f"**Page {chunk['page_num']}** (score: {chunk['score']})")
                        st.caption(chunk["text"][:200] + "...")
                        st.divider()

        st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.warning("Pehle Groq API key daalo")