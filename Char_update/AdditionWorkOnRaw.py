%%writefile chat2.py
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer
import numpy as np
import re

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

    def _load_chunks(self):
        """Load and preprocess text chunks from Ramayana more intelligently"""
        with open("ram_ji.txt", "r", encoding="utf-8") as f:
            text = f.read()
        sentences = re.split(r'[।॥]', text)
        chunks = []
    
        chunk = ""
        for sentence in sentences:
            if len(chunk.split()) + len(sentence.split()) < 80:
                chunk += sentence.strip() + " "
            else:
                chunks.append(self._clean_text(chunk))
                chunk = sentence.strip() + " "
        if chunk:
            chunks.append(self._clean_text(chunk))
        return [c for c in chunks if len(c.split()) > 3]

    def _clean_text(self, text):
        """Clean the text for readability while keeping the essence"""
        text = re.sub(r'[^\w\s।॥ॐ.,!?-]', '', text)  # Preserve meaningful characters
        return re.sub(r'\s+', ' ', text).strip()

    def _get_relevant_chunks(self, query, top_k=3):
         """Fetch most relevant text based on the query using cosine similarity"""
         chunks = self._load_chunks()
         query_embed = self.embedder.encode([query])
         chunk_embeds = self.embedder.encode(chunks)
     
         similarities = cosine_similarity(query_embed, chunk_embeds)[0]
         top_indices = np.argsort(similarities)[-top_k:]
         return [chunks[i] for i in reversed(top_indices)]


    def generate_response(self, query):
        """Generate response based on the spiritual depth of Shri Ram"""
        # Construct prompt with deeper context and character profile
        context = "\n".join(self._get_relevant_chunks(query))

        prompt = f"""
        As Lord Shri Ram, the Maryada Purushottam, answer this question while embodying:
        - Perfect adherence to dharma
        - Compassion even to adversaries
        - Unwavering commitment to truth
        - Respect for all beings
        - The wisdom of the Vedas
        
        Refer to these teachings from the Ramayana when relevant:
        {context}
        
        Question: {query}
        
        Answer as Shri Ram would - with clarity, humility and deep wisdom, in under 100 words.
        """


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

    ai = ShriRamAI()
    chat_history = []

    for query in test_queries:
        print(f"\nQuestion: {query}")
        print(ai.generate_response(query))

    # while True:
        # user_input = input("You: ").strip()
        # if user_input.lower() in ["exit", "quit"]:
        #     print("\nShri Ram:\n\nMay peace guide your path always. 🙏🏼\nॐ")
        #     break

        # # You can enable memory here by passing history to a new generate_with_history() method
        # try:
        #     response = ai.generate_response(user_input)
        #     chat_history.append(user_input)
        #     print(response + "\n")
        # except Exception as e:
        #     print("Shri Ram:\n\nForgive me, something went wrong while guiding you.\nॐ")
        #     print(f"(Debug info: {str(e)})\n")

