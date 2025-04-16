import json

# Load the JSON data
def load_ramayana_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)
    
# Keyword-based matching for simplicity
def find_best_match(user_input, data):
    user_input_lower = user_input.lower()
    for entry in data:
        if (user_input_lower in entry.get("text", "").lower() or 
            user_input_lower in entry.get("theme", "").lower() or 
            user_input_lower in entry.get("dharma", "").lower()):
            return entry
    return None

# Generate ShriRamAI response from the matched entry
def generate_response(user_input, data):
    entry = find_best_match(user_input, data)
    
    if entry:
        quote = entry.get('text', '').strip().splitlines()[0]
        response = f"""
        🚩 *ShriRamAI responds with divine wisdom:*

        🕉️ **Dharma:** {entry.get('dharma', 'N/A')}

        📖 **From the Ramayana:**  
        "{quote}..."

        🎭 *Theme:* {entry.get('theme', 'Unknown')}  
        ❤️ *Emotion:* {entry.get('emotion', 'Neutral')}  
        """
        return response.strip()
    else:
        return (
            "🙏 O seeker, your intention is pure. Yet, I find no verse that directly reflects your query.\n"
            "May you try again, for the truth shall reveal itself in due time."
        )

# Entry point
if __name__ == "__main__":
    print("🌟 Welcome to ShriRamAI 🌟")
    print("Ask your question (type 'exit' to quit):\n")
    
    ramayana_data = load_ramayana_data('ramayana.json')
    
    while True:
        user_input = input("🗣️ You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("🙏 ShriRamAI: May dharma guide your path. Farewell, noble soul.")
            break
        
        response = generate_response(user_input, ramayana_data)
        print(response)
        print("\n" + "-"*70 + "\n")