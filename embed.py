from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load the pre-trained model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Load the chunks from the file
with open("chunks.txt", "r") as file:
    chunks = file.readlines()

# Generate embeddings for each chunk
embeddings = model.encode(chunks)

# Create a FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])  # Using L2 distance
index.add(np.array(embeddings))

# Save the FAISS index
faiss.write_index(index, "ramayana_faiss.index")
