# 1. Load the Ramayana text file
with open("ramayana.txt", "r", encoding="utf-8") as file:
    full_text = file.read()

# 2. Extract Book I, Canto I – (Modify if needed based on actual format in your file)
import re

# Pattern assumes format like: "Canto I" or "CANTO I" — modify to match your exact file style
match = re.search(r"(Book I.*?)Canto I(.*?)Canto II", full_text, re.DOTALL | re.IGNORECASE)

if match:
    canto_i_text = "Canto I" + match.group(2).strip()
else:
    raise ValueError("Canto I not found. Please check formatting.")

# 3. Prepare the prompt
prompt = f"""
You are an expert in Indian epics and a skilled AI assistant for a spiritual chatbot project called "ShriRamAI."

Your task is to extract meaningful, structured information from the following Canto of the Ramayana. Please read the full passage carefully and provide the output in JSON format, following this structure:

{{
  "book": "",
  "canto": "",
  "text": "",
  "speakers": [],
  "emotions": [],
  "themes": [],
  "dharma": ""
}}

Rules:

- Use only insights that appear in the actual text.
- Summarize the dharma in a calm and spiritual tone.
- Identify characters only if they speak or are directly portrayed in action or emotion.
- The lists for emotions and themes must reflect what the passage reveals.
- Keep JSON formatting strict — no trailing commas, clean strings.

Now, here is the text from the Ramayana:

{canto_i_text}

Return ONLY the JSON object, with no explanation or extra output.
"""

# 4. Send this prompt to your Google LLM cell (wherever you're generating completions)
print(prompt)
