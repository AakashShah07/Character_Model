import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
from sentence_transformers import SentenceTransformer
from functools import lru_cache
import re
from typing import List

class RamayanaAI:
    def __init__(self):
        self._init_models()
        # Removed _setup_pipelines() call as it's not needed
        
    def _init_models(self):
        """Initialize all AI models with optimized settings"""
        # Use GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device set to use {self.device}")
        
        # Sentence embedding model
        print("Loading sentence transformer...")
        self.embedder = SentenceTransformer(
            'sentence-transformers/all-mpnet-base-v2',
            device=self.device
        )
        
        # Text generation model
        print("Loading GPT-2 model...")
        model_name = "gpt2"  # Using base GPT-2 for wider compatibility
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize text cleaner directly without separate pipeline setup
        self.text_cleaner = lambda x: x  # Placeholder - removed T5 cleaner for simplicity

    @lru_cache(maxsize=32)
    def _load_chunks(self) -> List[str]:
        """Cached text loader with preprocessing"""
        with open("ramayana.txt", "r", encoding="utf-8") as f:
            text = f.read()
            
        # Basic cleaning
        chunks = [
            self._clean_text(chunk) 
            for chunk in text.split("\n\n") 
            if chunk.strip()
        ]
        return chunks

    def _clean_text(self, text: str) -> str:
        """Basic text cleaning"""
        text = re.sub(r'[^\w\s।॥ॐ]', '', text)  # Remove special chars
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
        return text

    def _get_relevant_chunks(self, query: str, top_k: int = 3) -> List[str]:
        """Semantic search with caching"""
        chunks = self._load_chunks()
        query_embed = self.embedder.encode(query)
        chunk_embeds = self.embedder.encode(chunks)
        
        # Use cosine similarity
        scores = np.dot(chunk_embeds, query_embed) / (
            np.linalg.norm(chunk_embeds, axis=1) * np.linalg.norm(query_embed)
        )
        top_indices = np.argsort(scores)[-top_k:]
        return [chunks[i] for i in top_indices]

    def _generate_prompt(self, query: str, context: str) -> str:
        """Dynamic prompt engineering"""
        return f"""
        [CHARACTER] You are Lord Shri Ram, the seventh avatar of Vishnu.
        [KNOWLEDGE] You strictly follow Dharma and the teachings of Valmiki Ramayana.
        [STYLE] Answer with divine wisdom, compassion, and poetic elegance.
        [CONTEXT] {context}
        [QUESTION] {query}
        [ANSWER] श्री राम: 
        """

    def _postprocess_response(self, text: str) -> str:
        """Basic response cleaning"""
        # Remove prompt leakage
        text = text.split("[ANSWER]")[-1].strip()
        
        # Remove any remaining special tokens
        text = text.replace("[CHARACTER]", "").replace("[KNOWLEDGE]", "").replace("[STYLE]", "")
        
        # Ensure proper closing
        if not any(marker in text for marker in ["।", "॥", "ॐ"]):
            text += " ॥"
            
        return f"श्री राम: {text}\n\nॐ"

    def generate_response(self, query: str) -> str:
        """End-to-end response generation"""
        # Get relevant context
        context_chunks = self._get_relevant_chunks(query)
        context = "\n".join(context_chunks)
        
        # Generate prompt
        prompt = self._generate_prompt(query, context)
        
        # Generate response
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=200,
            num_beams=5,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            no_repeat_ngram_size=3,
            early_stopping=True,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Decode and clean
        raw_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._postprocess_response(raw_response)

# Example usage
if __name__ == "__main__":
    print("Initializing Ramayana AI...")
    ai = RamayanaAI()
    
    # Test query
    test_query = "How should one treat their enemies?"
    print(f"\nQuestion: {test_query}")
    print(ai.generate_response(test_query))