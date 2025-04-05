from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load the full Ramayana text
with open("ramayana.txt", "r") as file:
    ramayana_text = file.read()

# Split the text into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_text(ramayana_text)

# Save the chunks for later use
with open("chunks.txt", "w") as file:
    for chunk in chunks:
        file.write(f"{chunk}\n\n")
