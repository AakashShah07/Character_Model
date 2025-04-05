from transformers import pipeline, GPT2Tokenizer, GPT2LMHeadModel
from query_engine import get_relevant_chunks

# Load a pre-trained model and tokenizer
generator = pipeline('text-generation', model='gpt2')  # You can fine-tune your own model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Set the pad_token to eos_token to avoid padding issues
tokenizer.pad_token = tokenizer.eos_token

# Function to load chunks (adjust based on how your data is stored)
def load_chunks():
    with open("ramayana.txt", "r") as file:
        text = file.read()
    
    chunks = text.split("\n\n")  # Split the text into chunks by paragraphs

    print(f"Loaded chunks: {len(chunks)}")  # Debugging

    chunks = text.split("\n\n")  # Split the text into chunks by paragraphs
    return chunks

def generate_response(query):
    # Load chunks of the text
    chunks = load_chunks()
    
    # Get relevant chunk indices for the query
    relevant_chunk_indices = get_relevant_chunks(query, chunks)
    
    # Check if relevant chunks are found, otherwise use a fallback context
    if not relevant_chunk_indices:
        context = "Sorry, I don't have relevant information to answer this question."
    else:
        context = "\n\n".join([chunks[i] for i in relevant_chunk_indices])

    # Construct the prompt with a character role
    prompt = f"""
    You are Lord Shree Ram, a noble and righteous prince. Answer the following question with wisdom, grace, and compassion.

    Context:
    {context}

    User: {query}
    Shree Ram:
    """
    
    # Tokenize the input prompt and context
   # Tokenize the input prompt and context
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512, padding=True)

# Generate the response using the model with max_new_tokens
    outputs = model.generate(**inputs, max_new_tokens=150, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

# Decode the generated output
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Return the response
    return response


# Example usage
query = "How did Lord Ram treat his enemies?"
response = generate_response(query)
print(response)
