import re

# Step 1: Read original file
with open("ramayana.txt", "r", encoding="utf-8") as file:
    raw_text = file.read()

# Step 2: Split into paragraphs or stanzas
paragraphs = [p.strip() for p in raw_text.split("\n\n") if p.strip()]

# Step 3: Keywords Shri Ram would speak about
keywords = [
    "dharma", "truth", "enemy", "compassion", "duty", "sacrifice",
    "sita", "lakshman", "righteousness", "kingdom", "peace", "anger",
    "justice", "self-control", "forgiveness", "god", "lord", "divine",
    "devotion", "faith", "fate", "destiny", "virtue", "evil", "sin"
]

character_names = [
    "ram", "shri ram", "rama", "lakshman", "laxman", "bharat", "shatrughna",
    "sita", "hanuman", "ravana", "vibhishan", "kaikeyi", "dasharatha", "valmiki"
]

# Step 4: Clean and filter
cleaned = []
for para in paragraphs:
    # Optional: Normalize text
    para = re.sub(r'[^\w\s.,!?-]', '', para)
    para = re.sub(r'\s+', ' ', para).strip()

    # Check if paragraph is worth keeping
    if any(k in para.lower() for k in keywords + character_names) and len(para.split()) >= 5:
        cleaned.append(para)

# Step 5: Save cleaned content
with open("shriram_cleaned.txt", "w", encoding="utf-8") as out:
    for line in cleaned:
        out.write(line + "\n\n")

print(f"✅ Cleaned {len(cleaned)} paragraphs. Output saved to shriram_cleaned.txt.")


# Question: How should one treat their enemies?
# Shri Ram:

# Then if thy judgment still refers To Fate this plot of his and hers, My mind herein can neer agree And O, in this be ruled by me. Weak, void of manly pride are they Who bend to Fates imputed sway The choicest souls. the nobly great Disdain to bow their heads to Fatis. And he who dares his Fatia control With vigorous act and many soul, Though threatening Fatitis his hopes assail, Unmoved through all need never quail. This day mankind shall learn aright The power of Fatide and human might, So shall the gulf that lies between A man and Fatine be clearly seen. The might of fate subduered by him This hour the citizens shall see, Who saw its intervention stay Thy consecrating rites to-day. My power shall turn this Fatiloe away, That threatens as, with furious stride, An elephant who scorns...
# ॐ

# Question: Who is sita?
# Shri Ram:

# the goddess of Earth.
# ॐ