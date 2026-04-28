import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def load_index():
    with open("page_index.json", "r") as f:
        page_index = json.load(f)
    with open("page_contents.json", "r") as f:
        page_contents = json.load(f)
    with open("chunks.json", "r") as f:
        chunks = json.load(f)
    index = faiss.read_index("faiss_index.bin")
    return page_index, page_contents, chunks, index

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

def retrieve(query, top_pages=5, top_chunks=3):
    try:
        page_index, page_contents, chunks, index = load_index()
    except FileNotFoundError:
        print("❌ Index nahi mila — pehle ingest.py run karo")
        return []

    relevant_pages = search_page_index(query, page_index, top_n=top_pages)

    if not relevant_pages:
        relevant_pages = [int(k) for k in page_contents.keys()]

    print(f"📄 Page index ne ye pages suggest kiye: {relevant_pages}")

    # query embed karo
    query_embedding = model.encode([query]).astype("float32")
    
    # faiss search
    distances, indices = index.search(query_embedding, top_chunks * 3)

    # sirf relevant pages ke chunks lo
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(chunks):
            chunk = chunks[idx]
            if chunk["page_num"] in relevant_pages:
                results.append({
                    "text": chunk["text"],
                    "page_num": chunk["page_num"],
                    "score": round(1 / (1 + float(distances[0][i])), 3)
                })
        if len(results) >= top_chunks:
            break

    return results

if __name__ == "__main__":
    results = retrieve("What is CAP theorem?")
    for chunk in results:
        print(f"Page {chunk['page_num']} | Score: {chunk['score']}")
        print(chunk["text"][:200])
        print()