import faiss
import PyPDF2
import json
import numpy as np
from collections import defaultdict
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "are", "was", "were",
    "i", "my", "your", "his", "her", "we", "they", "it", "this", "that"
}

def extract_pages(pdf_path):
    pages = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text.strip():
                pages.append({"page_num": i + 1, "text": text.strip()})
    return pages

def build_page_index(pages):
    page_index = defaultdict(list)
    page_contents = {}
    for page in pages:
        page_num = page["page_num"]
        page_contents[str(page_num)] = page["text"]
        words = page["text"].lower().split()
        keywords = set([
            w.strip(".,;:()[]") for w in words
            if w.strip(".,;:()[]") not in STOP_WORDS and len(w) > 2
        ])
        for keyword in keywords:
            if page_num not in page_index[keyword]:
                page_index[keyword].append(page_num)
    return dict(page_index), page_contents

def make_chunks(pages, chunk_size=50, overlap=10):
    chunks = []
    chunk_id = 0
    for page in pages:
        words = page["text"].split()
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk_text = " ".join(words[start:end])
            chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "page_num": page["page_num"],
            })
            chunk_id += 1
            start += chunk_size - overlap
    return chunks

def index_resume(pdf_path="ASD.pdf"):
    print("📄 PDF padh raha hoon...")
    pages = extract_pages(pdf_path)
    print(f"   {len(pages)} pages mili")

    print("🗂️  Page index bana raha hoon...")
    page_index, page_contents = build_page_index(pages)
    with open("page_index.json", "w") as f:
        json.dump(page_index, f)
    with open("page_contents.json", "w") as f:
        json.dump(page_contents, f)
    print(f"   {len(page_index)} keywords index hue")

    print("✂️  Chunks bana raha hoon...")
    chunks = make_chunks(pages)
    with open("chunks.json", "w") as f:
        json.dump(chunks, f)
    print(f"   {len(chunks)} chunks bane")

    print("🔢 Embeddings bana raha hoon...")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True).astype("float32")

    print("💾 FAISS index bana raha hoon...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, "faiss_index.bin")

    print(f"\n✅ Done!")
    print(f"   Page index: {len(page_index)} keywords")
    print(f"   FAISS: {len(chunks)} chunks")

if __name__ == "__main__":
    index_resume()