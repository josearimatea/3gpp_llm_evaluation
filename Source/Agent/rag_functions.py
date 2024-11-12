import faiss
import torch
from sentence_transformers import SentenceTransformer
import pickle
import numpy as np

# Function to load the FAISS index
def load_faiss_index(index_file_path="../../Files/faiss_index.bin"):
    faiss_index = faiss.read_index(index_file_path)
    return faiss_index

# Function to perform search on the FAISS index
def search_faiss_index(query_text, faiss_index, model=None, top_k=5):
    if model is None:
        model = SentenceTransformer('all-mpnet-base-v2', device="cpu")
    
    query_embedding = model.encode(query_text, convert_to_tensor=True)
    query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=0).cpu().numpy().reshape(1, -1)
    
    distances, indices = faiss_index.search(query_embedding, top_k)
    return distances, indices

# Function to perform RAG (Retrieve and Generate) search
def search_RAG(query_text, index_file_path="../../Files/faiss_index.bin", chunks_path="../../Files/tspec_chunks_markdown.pkl", top_k=3):
    faiss_index = load_faiss_index(index_file_path)
    tspec_chunks = load_chunks(chunks_path)
    distances, indices = search_faiss_index(query_text, faiss_index, top_k=top_k)

    result_texts = []
    for i, idx in enumerate(indices[0]):
        result_texts.append(f"Information {i + 1}:\n{tspec_chunks[idx]['text']}\n")

    del tspec_chunks
    return "\n".join(result_texts)

# Function to load chunks from a file
def load_chunks(filename):
    with open(filename, 'rb') as f:
        chunks = pickle.load(f)
    return chunks
