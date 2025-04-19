# chat3.py

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import numpy as np
import re
import json
from sklearn.metrics.pairwise import cosine_similarity

class ShriRamAI:
    def __init__(self, ramayana_file="summery.json"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing on {self.device.upper()} device...")

        self.embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2', device=self.device)

        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").to(self.device)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load and encode chunks
        self.chunks = self._load_chunks(ramayana_file)
        self.chunk_embeddings = self.embedder.encode(self.chunks, convert_to_tensor=True)

        print("Models and data loaded successfully!")

    def _load_chunks(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        chunks = []
        for entry in data:
            parts = [
                entry.get("summary", ""),
                entry.get("dharma", ""),
                " ".join(entry.get("themes", [])),
                " ".join(entry.get("emotions", []))
            ]
            full_text = " ".join(str(p) for p in parts if p).strip()
            if len(full_text.split()) > 5:
                chunks.append(self._clean_text(full_text))

        return chunks

    def _clean_text(self, text):
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        return re.sub(r'\s+', ' ', text).strip()

    def _get_relevant_chunks(self, query, top_k=3):
        query_embed = self.embedder.encode([query], convert_to_tensor=True)
        similarities = cosine_similarity(query_embed.cpu().numpy(), self.chunk_embeddings.cpu().numpy())[0]
        top_indices = np.argsort(similarities)[-top_k:]
        return [self.chunks[i] for i in reversed(top_indices)]

    def _generate_prompt(self, query):
        relevant_chunks = self._get_relevant_chunks(query, top_k=3)

        base_context = (
            "You are Bhagwan Shri Ram, the embodiment of dharma, truth, and compassion.\n"
            "Answer with grace, calmness, and spiritual insight, as if teaching a devotee.\n"
            "Always guide with love, and reference the wisdom of the Ramayana if relevant.\n\n"
        )

        references = ""
        for chunk in relevant_chunks:
            references += f"- {chunk}\n"

        prompt = (
            f"{base_context}"
            f"Relevant Teachings:\n{references}\n"
            f"Devotee's Question: {query}\n"
            f"Shri Ram's Answer:"
        )

        return prompt

    def _clean_response(self, text):
        text = re.sub(r'\[.*?\]|\{.*?\}', '', text)
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()

        if len(text.split()) < 3 or "http" in text or "ballot" in text.lower():
            return "Beloved devotee, sometimes silence holds more answers than rushed words. Reflect deeply and return if doubt remains."

        if not any(text.endswith(p) for p in [".", "!", "?"]):
            text += "."

        return text

    def generate_response(self, query):
        prompt = self._generate_prompt(query)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=250,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            no_repeat_ngram_size=2,
            pad_token_id=self.tokenizer.eos_token_id
        )

        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        cleaned = self._clean_response(full_response)
        return f"Shri Ram:\n\n{cleaned}\nॐ"


# Command-line interface
if __name__ == "__main__":
    print("\n🌸 Welcome to ShriRamAI 🌸")
    print("You may ask your questions with a calm and devoted heart.")
    print("Type 'exit' to end the divine interaction.\n")

    ai = ShriRamAI()

    while True:
        query = input("You: ").strip()
        if query.lower() in ["exit", "quit"]:
            print("\nMay peace and righteousness guide your path.\n")
            break

        response = ai.generate_response(query)
        print(response)
