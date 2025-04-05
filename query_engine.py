def generate_response(query):
    # Assuming chunks is already loaded or defined elsewhere
    chunks = load_chunks()  # Replace this with the actual method to load chunks
    relevant_chunk_indices = get_relevant_chunks(query, chunks)
    
    # Now proceed with further logic to generate the response using the relevant chunks
    response = "Generated response based on relevant chunks"  # Replace with actual logic
    return response

def load_chunks():
    # Example function to load chunks of text (Ramayana)
    chunks = ["chunk1", "chunk2", "chunk3"]  # Replace with actual text splitting logic
    return chunks

def get_relevant_chunks(query, chunks):
    relevant_chunks = []
    
    # Define some terms that would likely be in relevant chunks for the given question
    keywords = ["enemy", "Ravana", "battle", "compassion", "righteousness", "forgiveness", "confrontation"]
    
    # Loop through each chunk to check if it contains any of the keywords
    for idx, chunk in enumerate(chunks):
        # Check if any of the keywords appear in the chunk
        if any(keyword.lower() in chunk.lower() for keyword in keywords):
            relevant_chunks.append(idx)
    
    return relevant_chunks


