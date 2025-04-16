%%writefile chat3.py

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import re
import json

class ShriRamAI:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing on {self.device.upper()} device...")

        # Embedding model for context understanding
        self.embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2', device=self.device)

        # Text generation model (Fine-tuned for spirituality)
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").to(self.device)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Models loaded successfully!")

        # Memory to store conversation history (simplified)
        self.conversation_history = []


    def _get_relevant_chunks(self, query, top_k=1):
        """Returns top_k most relevant chunks (dicts) based on semantic similarity."""
        query_embedding = self.embedder.encode(query, convert_to_tensor=True)
        chunk_embeddings = self.chunk_embeddings  # already precomputed embeddings

        # Compute cosine similarity
        scores = util.pytorch_cos_sim(query_embedding, chunk_embeddings)[0]

        # Get top_k results
        top_results = torch.topk(scores, k=top_k)

        # Return the actual chunk dicts, not just the text
        return [self.chunks[idx] for idx in top_results.indices]


    def _load_chunks(self, file_path):
        """Load and preprocess text chunks from Ramayana in JSON format"""
        with open("ramayana.json", "r", encoding="utf-8") as f:
            data = json.load(f)  # Load the JSON data as a list of dictionaries
        
        chunks = []
        chunk = ""
        
        # Extracting and processing the 'text' field from each dictionary
        for entry in data:
            text = entry.get("text", "")
            sentences = re.split(r'[।॥]', text)  # Split by sentence markers in Sanskrit
            
            for sentence in sentences:
                if len(chunk.split()) + len(sentence.split()) < 80:
                    chunk += sentence.strip() + " "
                else:
                    chunks.append(self._clean_text(chunk))
                    chunk = sentence.strip() + " "
            if chunk:
                chunks.append(self._clean_text(chunk))
        
        # Return chunks that have more than 3 words
        return [c for c in chunks if len(c.split()) > 3]


    def _clean_text(self, text):
        """Clean the text for readability while keeping the essence"""
        text = re.sub(r'[^\w\s।॥ॐ.,!?-]', '', text)  # Preserve meaningful characters
        return re.sub(r'\s+', ' ', text).strip()

    def _get_relevant_chunks(self, query, top_k=3, file_path="ramayana.json"):
         """Fetch most relevant text based on the query using cosine similarity"""
         chunks = self._load_chunks("file_path")
         query_embed = self.embedder.encode([query])
         chunk_embeds = self.embedder.encode(chunks)
     
         similarities = cosine_similarity(query_embed, chunk_embeds)[0]
         top_indices = np.argsort(similarities)[-top_k:]
         return [chunks[i] for i in reversed(top_indices)]

    def _generate_prompt(self, query):
        """Generate a detailed prompt with emotion, theme, and dharma."""
        relevant_chunks = self._get_relevant_chunks(query)

        prompt = ""
        for chunk in relevant_chunks:
            # If chunk is a string (fallback), wrap it in a dict
            if isinstance(chunk, str):
                chunk = {"text": chunk, "emotion": "", "theme": "", "dharma": ""}

            text = chunk.get("text", "")
            emotion = chunk.get("emotion", "No emotion available")
            theme = chunk.get("theme", "No theme available")
            dharma = chunk.get("dharma", "No dharma available")

            prompt += f"Emotion: {emotion}\nTheme: {theme}\nDharma: {dharma}\n\n"
            prompt += f"Text: {text}\n\n"

        prompt += f"Question: {query}\nAnswer:"
        return prompt


    def generate_response(self, query):
        """Generate response based on the spiritual depth of Shri Ram"""
        # Construct prompt with deeper context and character profile
        prompt = self._generate_prompt(query)

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            no_repeat_ngram_size=2,
            pad_token_id=self.tokenizer.eos_token_id
        )

        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        cleaned = self._clean_response(full_response)

        # Save the query and response to conversation history
        self.conversation_history.append(f"Q: {query}\nA: {cleaned}")

        return f"Shri Ram:\n\n{cleaned}\nॐ"



    def _clean_response(self, text):
        """Clean and post-process response"""
        text = re.sub(r'\[.*?\]|\{.*?\}', '', text)
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()

        if len(text.split()) < 3 or "http" in text or "ballot" in text.lower():
            return "I apologize, but I cannot provide a clear answer to that question."

        if not any(text.endswith(p) for p in [".", "!", "?"]):
            text += "."
        return text

# Example usage:
if __name__ == "__main__":
    print("\n🌸 Welcome to ShriRamAI 🌸")
    print("You may ask your questions with a calm and devoted heart.")
    print("Type 'exit' to end the divine interaction.\n")

    ai = ShriRamAI()
    test_queries = [
        "Why did you go to the forest?",
        "Tell me about Sita’s strength.",
        "What is true devotion?",
        "Who is Hanuman to you?",
        "How can one follow dharma?",
        "How do I find peace in pain?",
        "What happened after the war with Ravana?",
        "Why did you forgive your enemies?",
    ]

    for query in test_queries: 
        print(f"\nQuestion: {query}")
        print(ai.generate_response(query))
