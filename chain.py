from groq import Groq
from retriever import retrieve



def build_prompt(query, chunks):
    context = ""
    for chunk in chunks:
        context += f"[Page {chunk['page_num']}]\n{chunk['text']}\n\n"

    prompt = f"""You are a helpful study assistant for Advanced System Design.
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't have enough information."
Always mention which page the answer came from.

Context:
{context}

Question: {query}

Answer:"""
    return prompt

def ask(query):
    client = Groq()
    print(f"\n🔍 Query: {query}")
    chunks = retrieve(query)

    if not chunks:
        return "No relevant content found."

    prompt = build_prompt(query, chunks)
    print("🤖 Soch raha hoon...\n")

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content

    print("=" * 50)
    print(answer)
    print("\n📄 Sources:")
    for chunk in chunks:
        print(f"   Page {chunk['page_num']} (score: {chunk['score']})")
    print("=" * 50)

    return answer

if __name__ == "__main__":
    ask("What is CAP theorem?")