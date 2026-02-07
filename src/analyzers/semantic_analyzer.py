import google.generativeai as genai
import pandas as pd
import time
import json
import re
import os

# --- CONFIGURATION ---
API_KEY = os.getenv("GOOGLE_API_KEY")
INPUT_FILENAME = "outputs/intermediate_sentences.json"
OUTPUT_FILENAME = "outputs/thesis_semantic_data_final.csv"

if not API_KEY:
    raise RuntimeError("Missing GOOGLE_API_KEY environment variable.")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

# --- REFINED PROMPT ---
# We now ask for a simple list of strings to avoid syntax errors
SYSTEM_PROMPT = """
You are a semantic analyzer. I will provide a list of sentences containing a target word.
Classify the usage of the target word in each sentence into exactly one category:
1. "LITERAL" (Physical/Core meaning)
2. "IDIOMATIC" (De-lexicalized, Metaphor, or Fixed Phrase)

Return a strict JSON ARRAY of strings corresponding to the order of the input sentences.
Example Input: ["I run fast", "I run a business"]
Example Output: ["LITERAL", "IDIOMATIC"]
DO NOT return the original sentences. Just the tags.
"""

def load_sentences(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f" Error loading input file: {e}")
        return None

def get_tags_from_api(word, sentences):
    """
    Sends sentences to API and retrieves a list of tags (LITERAL/IDIOMATIC).
    """
    prompt = f"""
    Target Word: "{word}"
    Sentences to classify:
    {json.dumps(sentences)}
    """
    
    try:
        response = model.generate_content(
            contents=[{"role": "user", "parts": [{"text": SYSTEM_PROMPT + "\n" + prompt}]}]
        )
        
        raw_text = response.text
        
        # 1. Cleaning: Find the JSON array using Regex (Ignores 'Here is the json...' text)
        match = re.search(r'\[.*\]', raw_text, re.DOTALL)
        if not match:
            print(f" No JSON array found for '{word}'. Raw output: {raw_text[:50]}...")
            return ["ERROR"] * len(sentences)
            
        json_str = match.group(0)
        
        # 2. Parsing
        tags = json.loads(json_str)
        
        # 3. Validation: Ensure we got the right number of tags
        if len(tags) != len(sentences):
            print(f" Mismatch for '{word}': Sent {len(sentences)}, got {len(tags)} tags.")
            return ["ERROR"] * len(sentences)
            
        # 4. Standardization: Ensure tags are uppercase strings
        return [str(tag).upper() for tag in tags]

    except Exception as e:
        print(f" API Exception for '{word}': {e}")
        return ["ERROR"] * len(sentences)

def main():
    # 1. Load Data
    data = load_sentences(INPUT_FILENAME)
    if not data: return

    # 2. Group data by Lemma
    # Structure: { "run": [ {record1}, {record2}, {record3} ] }
    grouped_data = {}
    for item in data:
        lemma = item['lemma']
        if lemma not in grouped_data:
            grouped_data[lemma] = []
        grouped_data[lemma].append(item)

    final_rows = []
    sorted_words = sorted(grouped_data.keys())
    total = len(sorted_words)

    print(f"--- STARTING ROBUST ANALYSIS FOR {total} WORDS ---")

    # 3. Process
    for i, word in enumerate(sorted_words):
        records = grouped_data[word]
        # Extract just the sentence strings for the API
        sentences_to_send = [r['sentence'] for r in records]
        
        print(f"[{i+1}/{total}]  Tagging '{word}' ({len(sentences_to_send)} sentences)...")
        
        # Get the list of tags (e.g., ['LITERAL', 'IDIOMATIC', 'LITERAL'])
        tags = get_tags_from_api(word, sentences_to_send)
        
        # 4. Merge tags back into the records
        for record, tag in zip(records, tags):
            row = {
                "Lemma": record['lemma'],
                "Register": record['register'],
                "Mood": record['mood'],
                "Usage_Category": tag, # The extracted tag
                "Full_Sentence": record['sentence']
            }
            final_rows.append(row)
        
        time.sleep(1.0) # Respect rate limits

    # 5. Save
    df = pd.DataFrame(final_rows)
    df.to_csv(OUTPUT_FILENAME, index=False)
    
    print("\n" + "="*40)
    print(f" DONE. Saved to {OUTPUT_FILENAME}")
    print("Summary of Results:")
    print(df['Usage_Category'].value_counts())
    print("="*40)

if __name__ == "__main__":
    main()