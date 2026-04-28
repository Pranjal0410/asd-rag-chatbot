import chromadb
import json
from sentence_transformers import SentenceTransformer

# embedding model — local, free
model = SentenceTransformer("all-MiniLM-L6-v2")

def load_index():
    with open("page_index.json", "r") as f:
        page_index = json.load(f)
    with open("page_contents.json", "r") as f:
        page_contents = json.load(f)
    return page_index, page_contents

def search_page_index(query, page_index, top_n=5):
    query_words = query.lower().split()
    page_scores = {}

    for word in query_words:
        word = word.strip("?,.")
        if word in page_index:
            for page_num in page_index[word]:
                page_scores[page_num] = page_scores.get(page_num, 0) + 1

    ranked_pages = sorted(page_scores.items(), key=lambda x: x[1], reverse=True)
    return [p[0] for p in ranked_pages[:top_n]]

def search_vectors(query, relevant_pages, n_results=3):
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection("resume")

    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where={"page_num": {"$in": [int(p) for p in relevant_pages]}}
    )

    chunks = []
    for i, doc in enumerate(results["documents"][0]):
        chunks.append({
            "text": doc,
            "page_num": results["metadatas"][0][i]["page_num"],
            "score": round(1 / (1 + results["distances"][0][i]), 3)
        })

    return chunks

def retrieve(query, top_pages=5, top_chunks=3):
    try:
        page_index, page_contents = load_index()
    except FileNotFoundError:
        print("❌ page_index.json nahi mila — pehle ingest.py run karo")
        return []

    relevant_pages = search_page_index(query, page_index, top_n=top_pages)

    if not relevant_pages:
        print("⚠️  Page index mein koi match nahi, full search kar raha hoon...")
        relevant_pages = [int(k) for k in page_contents.keys()]

    print(f"📄 Page index ne ye pages suggest kiye: {relevant_pages}")
    chunks = search_vectors(query, relevant_pages, n_results=top_chunks)
    return chunks