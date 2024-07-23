import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load an embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# List of string texts
texts = ["Hello world", "Faiss is a library for efficient similarity search", "Embedding example"]

# Generate embeddings
embeddings = model.encode(texts)

# Create a dictionary to map embeddings to texts
embedding_to_text = {i: text for i, text in enumerate(texts)}

# Convert embeddings to float32 (required by Faiss)
embeddings = np.array(embeddings).astype('float32')

# Initialize a Faiss index
index = faiss.IndexFlatL2(embeddings.shape[1])

# Add embeddings to the index
index.add(embeddings)

# Save the index and the embedding_to_text mapping
faiss.write_index(index, 'faiss_index.index')
with open('embedding_to_text.npy', 'wb') as f:
    np.save(f, embedding_to_text)

print("Index and text mapping saved.")




# Load the index and the embedding-to-text mapping
index = faiss.read_index('faiss_index.index')
with open('embedding_to_text.npy', 'rb') as f:
    embedding_to_text = np.load(f, allow_pickle=True).item()

# Query text
query_text = "Find similar text"
query_embedding = model.encode([query_text]).astype('float32')

# Search the index
D, I = index.search(query_embedding, k=5)  # k is the number of nearest neighbors

# Retrieve and print the corresponding texts
for idx in I[0]:
    print(embedding_to_text[idx])
