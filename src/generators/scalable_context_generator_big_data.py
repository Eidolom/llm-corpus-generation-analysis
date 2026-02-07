import os
import google.generativeai as genai
import time
import json
import re

# --- CONFIGURATION ---
API_KEY = os.getenv("GOOGLE_API_KEY")
TARGET_WORDS_FILE = "data/target_words.txt"
OUTPUT_FILENAME = "outputs/intermediate_sentences.json"

# --- GENERATION SETTINGS ---
# 5 Batches x 10 sentences = 50 sentences per register (150 total per word)
NUM_BATCHES = 5 
SENTENCES_PER_REG_PER_BATCH = 10 

if not API_KEY:
    raise RuntimeError("Missing GOOGLE_API_KEY environment variable.")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash') # 2.5-flash is best for high-volume tasks

SYSTEM_PROMPT = """
You are a linguistic expert specializing in pragmatics and corpus generation.
Your task is to generate distinct, high-quality sentences for a target word based on strict register constraints.
The sentences must be appropriate for a CEFR B1/B2 learner.
"""

def clean_and_load_json(raw_text):
    """Clean markdown and parse JSON."""
    if not raw_text: return None
    try:
        # Regex to find the first [ and the last ]
        match = re.search(r'\[.*\]', raw_text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        else:
            # Fallback for simple strip
            cleaned = raw_text.replace('```json', '').replace('```', '').strip()
            return json.loads(cleaned)
    except Exception as e:
        print(f"   JSON Parse Error: {e}")
        return None

def load_target_words(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f" Error: '{filename}' not found.")
        return []

def generate_batch(word, batch_num):
    """Generates a single batch of 30 sentences (10 High, 10 Low, 10 Neutral)."""
    
    prompt = f"""
    Target Word: "{word}"
    Batch ID: {batch_num} (Ensure these sentences are unique from generic examples)

    Generate a JSON array containing exactly {3 * SENTENCES_PER_REG_PER_BATCH} sentences:
    
    1. {SENTENCES_PER_REG_PER_BATCH} sentences: Register HIGH (Formal), Mood Question. Context: Workplace/Professional.
    2. {SENTENCES_PER_REG_PER_BATCH} sentences: Register LOW (Casual), Mood Statement. Context: Friends/Personal/Slang.
    3. {SENTENCES_PER_REG_PER_BATCH} sentences: Register NEUTRAL (Direct), Mood Imperative. Context: Instructions/Navigation/Rules.

    Vary the sentence structure and vocabulary usage.
    Output JSON keys: "register", "mood", "sentence".
    """
    
    try:
        response = model.generate_content(
            contents=[{"role": "user", "parts": [{"text": SYSTEM_PROMPT + "\n" + prompt}]}]
        )
        return clean_and_load_json(response.text)
    except Exception as e:
        print(f"   API Error: {e}")
        return None

def main():
    words = load_target_words(TARGET_WORDS_FILE)
    if not words: return

    all_data = []
    total_words = len(words)
    
    print(f"--- STARTING GENERATION: {total_words} Words x {NUM_BATCHES} Batches ---")
    print(f"Target: {NUM_BATCHES * SENTENCES_PER_REG_PER_BATCH} sentences per register per word.")

    for i, word in enumerate(words):
        word_sentences = []
        print(f"\n[{i+1}/{total_words}]  Processing '{word}'...")
        
        for batch_idx in range(NUM_BATCHES):
            print(f"   ↳ Generating Batch {batch_idx+1}/{NUM_BATCHES}...", end="", flush=True)
            
            batch_data = generate_batch(word, batch_idx)
            
            if batch_data:
                # Validate we actually got a list
                if isinstance(batch_data, list):
                    # Add metadata
                    for item in batch_data:
                        item['lemma'] = word
                        item['Source'] = "Synthetic_V2"
                        item['CEFR_Target'] = "B1/B2"
                        # Normalize register names just in case
                        if 'register' in item:
                            item['register'] = item['register'].upper()
                    
                    word_sentences.extend(batch_data)
                    print(f"  Got {len(batch_data)} sentences.")
                else:
                    print(f"  format error (not a list).")
            else:
                print("  Failed.")
            
            # Sleep slightly longer to avoid rate limits on the loop
            time.sleep(2.0)

        all_data.extend(word_sentences)
        print(f"    Total for '{word}': {len(word_sentences)} sentences.")

    # Save
    if all_data:
        with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, indent=4)
        print(f"\nSUCCESS! Generated {len(all_data)} total sentences.")
        print(f"Saved to {OUTPUT_FILENAME}")
    else:
        print("\n No data generated.")

if __name__ == "__main__":
    main()