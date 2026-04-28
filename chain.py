from groq import Groq
from retriever import retrieve
from memory import save_conversation, get_relevant_memory

def build_prompt(query, chunks, memory_context=""):
    context = ""
    for chunk in chunks:
        context += f"[Page {chunk['page_num']}]\n{chunk['text']}\n\n"

    memory_section = ""
    if memory_context:
        memory_section = f"""Previous Conversations:
{memory_context}

"""

    prompt = f"""You are a helpful study assistant for Advanced System Design.
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't have enough information."
Always mention which page the answer came from.

{memory_section}Context:
{context}

Question: {query}

Answer:"""
    return prompt

def ask(query):
    client = Groq()
    print(f"\n🔍 Query: {query}")
    
    chunks = retrieve(query)
    memory_context = get_relevant_memory(query)
    
    if memory_context:
        print(f"🧠 Memory mili: {memory_context[:100]}...")

    if not chunks:
        return "No relevant content found."

    prompt = build_prompt(query, chunks, memory_context)
    print("🤖 Soch raha hoon...\n")

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content
    
    # conversation save karo memory mein
    save_conversation(query, answer)

    print("=" * 50)
    print(answer)
    print("\n📄 Sources:")
    for chunk in chunks:
        print(f"   Page {chunk['page_num']} (score: {chunk['score']})")
    print("=" * 50)

    return answer

if __name__ == "__main__":
    ask("What is CAP theorem?")