import supermemory
import os

client = supermemory.Supermemory(
    api_key=os.environ.get("SUPERMEMORY_API_KEY")
)

def save_conversation(question, answer):
    try:
        client.add(
            content=f"Q: {question}\nA: {answer}",
        )
        print("🧠 Memory saved!")
    except Exception as e:
        print(f"Memory save error: {e}")

def get_relevant_memory(query):
    try:
        results = client.search.documents(q=query)
        if results and results.results:
            memory_text = ""
            for r in results.results:
                if r.chunks:
                    memory_text += f"{r.chunks[0].content}\n\n"
            return memory_text.strip()
    except Exception as e:
        print(f"Memory search error: {e}")
    return ""