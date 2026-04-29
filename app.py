import streamlit as st
import os
from chain import ask, build_prompt
from retriever import retrieve
from memory import save_conversation, get_relevant_memory
from guardrails.input_guard import check_input
from guardrails.context_guard import check_chunks
from guardrails.output_guard import check_output
from groq import Groq
from tavily import TavilyClient

st.set_page_config(page_title="ASD Chatbot", page_icon="📚", layout="centered")

st.title("📚 Advanced System Design — Smart Chatbot")
st.caption("PDF knowledge + Web fallback + Guardrails + Memory")

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown("**Stack:**")
    st.markdown("- 🔍 FAISS + Page Index")
    st.markdown("- 🤖 Groq (llama-3.3-70b)")
    st.markdown("- 🧠 Supermemory")
    st.markdown("- 🛡️ ACF-style Guardrails")
    st.markdown("- 🌐 Tavily Web Fallback")
    st.divider()
    st.markdown("**Guardrail Decisions:**")
    st.markdown("✅ ALLOW — safe")
    st.markdown("⚠️ SANITISE — web fallback")
    st.markdown("🚫 BLOCK — blocked")

# ── SESSION STATE ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── CHAT HISTORY DISPLAY ──────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ── CHAT INPUT ────────────────────────────────────────────────────────────────
if question := st.chat_input("Kuch bhi poochho..."):

    with st.chat_message("user"):
        st.write(question)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):

        # ── GUARDRAIL 1: INPUT ────────────────────────────────────────────────
        input_result = check_input(question, chat_history=st.session_state.messages)

        if input_result["decision"] == "BLOCK":
            st.error("🚫 Blocked — This query violated safety rules.")
            st.caption(f"Score: {input_result['score']} | Signals: {input_result['signals']}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": "⛔ This query has been blocked by the safety guardrail."
            })
            st.stop()

        # guardrail badge
        if input_result["decision"] == "ALLOW":
            st.success(f"🛡️ Input: ALLOW (score: {input_result['score']})")
        else:
            st.warning(f"🛡️ Input: SANITISE — Web search mode (score: {input_result['score']})")

        web_context = ""
        chunks = []
        source_label = ""

        with st.spinner("Soch raha hoon..."):

            # ── OFF TOPIC → WEB SEARCH ────────────────────────────────────────
            if input_result["decision"] == "SANITISE" and "off_topic" in input_result["signals"]:
                source_label = "🌐 Web Search"
                tavily = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))
                try:
                    results = tavily.search(query=question, max_results=3)
                    for r in results["results"]:
                        web_context += f"{r['title']}\n{r['content'][:300]}\n\n"
                except Exception as e:
                    st.error(f"Web search error: {e}")

            else:
                # ── GUARDRAIL 2: CONTEXT ──────────────────────────────────────
                source_label = "📄 PDF"
                raw_chunks = retrieve(question)
                chunks = check_chunks(raw_chunks)

                sanitised = [c for c in chunks if c.get("sanitised")]
                if sanitised:
                    st.warning(f"⚠️ {len(sanitised)} chunk(s) sanitised")

                # low relevance → web fallback
                if not chunks or all(c["score"] < 0.3 for c in chunks):
                    source_label = "🌐 Web Fallback"
                    tavily = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))
                    try:
                        results = tavily.search(query=question, max_results=3)
                        for r in results["results"]:
                            web_context += f"{r['title']}\n{r['content'][:300]}\n\n"
                    except Exception as e:
                        st.error(f"Web search error: {e}")

            # ── MEMORY ────────────────────────────────────────────────────────
            memory_context = get_relevant_memory(question)

            # ── LLM CALL ──────────────────────────────────────────────────────
            prompt = build_prompt(
                question,
                chunks,
                memory_context,
                st.session_state.messages,
                web_context
            )

            groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}]
            )
            answer = response.choices[0].message.content

            # ── GUARDRAIL 3: OUTPUT ───────────────────────────────────────────
            output_result = check_output(answer, chunks)

        # ── DISPLAY ANSWER ────────────────────────────────────────────────────
        if output_result["decision"] == "BLOCK":
            st.error("🚫 Output blocked by safety guardrail.")
        else:
            if output_result["decision"] == "SANITISE":
                st.warning(f"⚠️ Output: {output_result['signals']}")
            
            st.write(output_result["answer"])
            st.caption(f"Source: {source_label} | Output score: {output_result['score']}")

            # sources
            if chunks:
                with st.expander("📄 Sources dekho"):
                    for chunk in chunks:
                        st.markdown(f"**Page {chunk['page_num']}** (score: {chunk['score']})")
                        st.caption(chunk["text"][:200] + "...")
                        st.divider()

        # ── SAVE MEMORY ───────────────────────────────────────────────────────
        save_conversation(question, output_result["answer"])
        st.session_state.messages.append({
            "role": "assistant",
            "content": output_result["answer"]
        })