import os
import google.generativeai as genai
import pandas as pd
import time
import json
import re
import math

# --- CONFIGURATION ---
API_KEY = os.getenv("GOOGLE_API_KEY")
INPUT_FILENAME = "outputs/intermediate_sentences.json"
OUTPUT_FILENAME = "outputs/thesis_semantic_data_final.csv"

# BATCH SIZE FOR ANALYSIS
# We will process 20 sentences at a time. This is the "Safe Zone" for LLM counting.
CHUNK_SIZE = 20 

if not API_KEY:
    raise RuntimeError("Missing GOOGLE_API_KEY environment variable.")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

SYSTEM_PROMPT = """
You are a semantic analyzer. I will provide a list of sentences containing a target word.
Classify the usage of the target word in each sentence into exactly one category:
1. "LITERAL" (Physical/Core meaning).
2. "IDIOMATIC" (De-lexicalized, Metaphor, Phrasal Verb, or Fixed Phrase).

Return a strict JSON ARRAY of strings.
Example Output: ["LITERAL", "IDIOMATIC", "LITERAL"]
"""

def load_sentences(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f" Error loading input file: {e}")
        return None

def get_tags_for_chunk(word, chunk_sentences, retry_count=0):
    """
    Analyzes a small chunk of sentences.
    """
    prompt = f"""
    Target Word: "{word}"
    Sentences to classify ({len(chunk_sentences)} items):
    {json.dumps(chunk_sentences)}
    
    Return exactly {len(chunk_sentences)} tags in a JSON array.
    """
    
    try:
        response = model.generate_content(
            contents=[{"role": "user", "parts": [{"text": SYSTEM_PROMPT + "\n" + prompt}]}]
        )
        
        raw_text = response.text.strip()
        
        # Cleaner Regex to find JSON array
        match = re.search(r'\[.*?\]', raw_text, re.DOTALL)
        if not match:
            # If regex fails, try parsing raw text if it looks like a list
            if raw_text.startswith('[') and raw_text.endswith(']'):
                json_str = raw_text
            else:
                raise ValueError("No JSON array found")
        else:
            json_str = match.group(0)
            
        tags = json.loads(json_str)
        
        # Validate Count
        if len(tags) != len(chunk_sentences):
            # Retry once if count is wrong
            if retry_count < 1:
                print(f"     Count mismatch ({len(tags)} vs {len(chunk_sentences)}). Retrying...")
                time.sleep(1)
                return get_tags_for_chunk(word, chunk_sentences, retry_count=1)
            else:
                print(f"     Failed chunk for '{word}'. Expected {len(chunk_sentences)}, got {len(tags)}.")
                return ["ERROR"] * len(chunk_sentences)

        return [str(t).upper() for t in tags]

    except Exception as e:
        print(f"     Error: {e}")
        return ["ERROR"] * len(chunk_sentences)

def main():
    data = load_sentences(INPUT_FILENAME)
    if not data: return

    # Group by Lemma
    grouped_data = {}
    for item in data:
        lemma = item['lemma']
        if lemma not in grouped_data: grouped_data[lemma] = []
        grouped_data[lemma].append(item)

    final_rows = []
    sorted_words = sorted(grouped_data.keys())
    total_words = len(sorted_words)

    print(f"--- STARTING CHUNKED ANALYSIS FOR {total_words} WORDS ---")

    for i, word in enumerate(sorted_words):
        records = grouped_data[word]
        total_items = len(records)
        print(f"[{i+1}/{total_words}]  Analyzing '{word}' ({total_items} sentences)...")
        
        # --- THE CHUNKING LOOP ---
        word_tags = []
        num_chunks = math.ceil(total_items / CHUNK_SIZE)
        
        for batch_idx in range(num_chunks):
            start = batch_idx * CHUNK_SIZE
            end = start + CHUNK_SIZE
            chunk_records = records[start:end]
            chunk_sentences = [r['sentence'] for r in chunk_records]
            
            # Get tags for just this small batch
            tags = get_tags_for_chunk(word, chunk_sentences)
            word_tags.extend(tags)
            
            # print(f"    Batch {batch_idx+1}/{num_chunks}: Processed {len(tags)} items.")
            time.sleep(0.5) # Fast sleep for small chunks

        # Merge tags back
        for record, tag in zip(records, word_tags):
            row = {
                "Lemma": record['lemma'],
                "Register": record['register'],
                "Mood": record['mood'],
                "Usage_Category": tag,
                "Full_Sentence": record['sentence']
            }
            final_rows.append(row)

    # Save
    df = pd.DataFrame(final_rows)
    df.to_csv(OUTPUT_FILENAME, index=False)
    
    print("\n" + "="*40)
    print(f" DONE. Saved to {OUTPUT_FILENAME}")
    print("Summary of Results:")
    print(df['Usage_Category'].value_counts())
    print("="*40)

if __name__ == "__main__":
    main()