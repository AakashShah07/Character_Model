import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from functools import lru_cache
import re


class ShriRamAI:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing on {self.device.upper()} device...")

        # Embedding model
        self.embedder = SentenceTransformer(
            'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
            device=self.device
        )

        # Text generation model
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").to(self.device)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Models loaded successfully!")

    @lru_cache(maxsize=32)
    def _load_chunks(self):
        """Load and preprocess text chunks from Ramayana"""
        with open("ram_ji.txt", "r", encoding="utf-8") as f:
            text = f.read()
        return [self._clean_text(chunk) for chunk in text.split("\n\n") if chunk.strip()]

    def _clean_text(self, text):
        """Clean text while preserving Hindi characters"""
        text = re.sub(r'[^\w\s।॥ॐ.,!?-]', '', text)
        return re.sub(r'\s+', ' ', text).strip()

    def get_relevant_context(query):
        # Extract relevant sections from Ramayana (a very simple example using keyword matching)
        relevant_chunks = []
        for section in ramayana_text:
            if any(keyword in section.lower() for keyword in query.lower().split()):
                relevant_chunks.append(section)
        return "\n".join(relevant_chunks)


    def _get_relevant_chunks(self, query, top_k=3):
        """Get top K relevant text chunks based on semantic similarity"""
        chunks = self._load_chunks()
        query_embed = self.embedder.encode(query)
        chunk_embeds = self.embedder.encode(chunks)
        scores = np.dot(chunk_embeds, query_embed)
        top_indices = np.argsort(scores)[-top_k:]
        return [chunks[i] for i in reversed(top_indices)]

    def generate_response(self, query):
      
      """Generate spiritually wise answer in the tone of Shri Ram"""

      # Fallback mini-knowledge base for key characters
      fallback_knowledge = {
          "laxman": "Laxman is my beloved younger brother. His devotion, courage, and loyalty are unmatched. He chose to walk the path of exile with me, guarding Sita and me with unwavering love.",
          "lakshman": "Laxman is my beloved younger brother. His devotion, courage, and loyalty are unmatched. He chose to walk the path of exile with me, guarding Sita and me with unwavering love.",
          "sita": "Sita is my beloved wife, a reflection of virtue and strength. Her grace, resilience, and unwavering faith illuminate the path of righteousness for all.",
          "hanuman": "Hanuman is my devoted companion, a symbol of strength and humility. His heart is pure, his service selfless. In him, devotion finds its highest form.",
          "bharat": "Bharat is my noble brother who rules in my name with great humility. Though he sat on the throne, his heart remained with me in the forest.",
          "ravana": "Ravana, though mighty, let pride cloud his wisdom. He fell not because of my strength, but because of his refusal to uphold dharma.",
      }

      # Clean and normalize query
      fallback_query = query.lower().strip()

      # Check if query matches fallback keys
      for key in fallback_knowledge:
          if key in fallback_query:
              return f"Shri Ram:\n\n{fallback_knowledge[key]}\nॐ"

      # Get context from the Ramayana text
      context = "\n".join(self._get_relevant_chunks(query))

      # Prompt tuned for spiritual clarity and modern language
      prompt = f"""
      You are Lord Shri Ram, the ideal king and embodiment of righteousness.
      Respond to the question as Shri Ram would—calm, clear, spiritually wise, and concise.
      Speak only in modern English. Avoid poetic or verse-like responses. Avoid ancient phrases.

      Context (from Ramayana): {context}

      Question: {query}

      Answer in clear, composed, spiritually insightful English, under 80 words, as Shri Ram.
      """

      # Tokenize and generate response using the model
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


    def _split_bilingual(self, text):
        """Split full response into Hindi + English parts"""
        match = re.search(r"([\u0900-\u097F\s।॥]+)[\n.:!?]*\s*(.*)", text.strip(), re.DOTALL)
        if match:
            return match.group(1).strip(), match.group(2).strip()
        return text.strip(), ""

    def _clean_response(self, text):
        text = re.sub(r'\[.*?\]|\{.*?\}', '', text)
        text = re.sub(r'[^a-zA-Z0-9\s.,!?-]', '', text) 
        text = re.sub(r'\n+', ' ', text).strip()

        if len(text.split()) < 3 or "http" in text or "ballot" in text.lower():
          return "I apologize, but I cannot provide a clear answer to that question."

        if not any(text.endswith(p) for p in [".", "!", "?"]):
          text += "."
        return text


# Example usage
if __name__ == "__main__":
    print("Initializing Shri Ram AI...")
    ai = ShriRamAI()

    test_queries = [
        "Who is father of sita?"
    ]

    for query in test_queries:
        print(f"\nQuestion: {query}")
        print(ai.generate_response(query))
