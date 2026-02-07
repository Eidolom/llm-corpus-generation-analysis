import google.generativeai as genai
import pandas as pd
import time
import json
import os

# --- CONFIGURATION ---
API_KEY = os.getenv("GOOGLE_API_KEY")
INPUT_FILENAME = "outputs/intermediate_sentences.json"
OUTPUT_FILENAME = "outputs/thesis_pragmatic_data_with_pos.csv"

if not API_KEY:
    raise RuntimeError("Missing GOOGLE_API_KEY environment variable.")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

# The "POS Tagging" Prompt - MODIFIED to explicitly request JSON in the output
SYSTEM_PROMPT_POSTAG = """
You are an expert linguistic analysis tool.
Your task is to analyze the provided sentences and determine the Part-of-Speech tag (e.g., V, N, Adj) for the specified target word, based on its usage in the sentence.
Your response MUST be a strict JSON array that follows the schema below. Output ONLY the JSON array, with NO extra text or markdown formatting (e.g., no ```json).
"""

# JSON Schema for POS Tagging Output (kept for prompt instruction)
POS_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "sentence": {"type": "STRING", "description": "The original sentence text."},
            "posTag": {"type": "STRING", "description": "The Part-of-Speech tag for the target word (V, N, Adj, etc.)."}
        },
        "required": ["sentence", "posTag"]
    }
}

def load_sentences(filename):
    """Loads sentences from the intermediate JSON file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f" Error: Input file '{filename}' not found. Please run 'pragmatic_generator.py' first.")
        return None
    except json.JSONDecodeError as e:
        print(f" Error: Could not parse JSON from '{filename}': {e}")
        return None

def generate_pos_tags(word, sentences_data):
    """Generates POS tags for the target word within the provided sentences."""
    
    # Filter sentences specific to the current word
    word_sentences = [item['sentence'] for item in sentences_data if item['lemma'] == word]
    
    if not word_sentences:
        return None

    # print(f"  Tagging POS for: '{word}' ({len(word_sentences)} sentences)...")
    
    # Include the schema definition in the prompt as text
    schema_definition = json.dumps(POS_SCHEMA, indent=2)

    prompt = f"""
    Target Word: {word}
    Sentences to Tag: {json.dumps(word_sentences)}
    Required Schema: {schema_definition}
    """
    
    try:
        # Call model.generate_content without the incompatible structured output parameters.
        response = model.generate_content(
            contents=[{"role": "user", "parts": [{"text": SYSTEM_PROMPT_POSTAG + "\n\n" + prompt}]}]
        )
        
        # The result.text should be the strict JSON array
        return json.loads(response.text)
        
    except Exception as e:
        # print(f" Error processing POS tags for '{word}': {e}")
        return None

def main():
    if not API_KEY:
        print(" ERROR: Please set GOOGLE_API_KEY in your environment.")
        return

    # 1. Load intermediate data
    all_sentences = load_sentences(INPUT_FILENAME)
    if not all_sentences:
        return

    # 2. Get unique words to tag
    target_words = sorted(list(set(item['lemma'] for item in all_sentences)))
    total_words = len(target_words)
    
    pos_lookup = {}
    
    # 3. Generate POS tags for each word group
    print(f"--- STARTING PART-OF-SPEECH TAGGING FOR {total_words} WORDS ---")
    for i, word in enumerate(target_words):
        print(f"[{i+1}/{total_words}]  Tagging POS for: '{word}'...")
        
        pos_data = generate_pos_tags(word, all_sentences)
        
        if pos_data:
            # Populate lookup table: {sentence_text: POS_TAG}
            for item in pos_data:
                if 'sentence' in item and 'posTag' in item:
                    pos_lookup[item['sentence']] = item['posTag'].upper()
        else:
            print(f"[{i+1}/{total_words}] Failed to tag POS for '{word}'. Skipping.")
        
        time.sleep(1.5) # Maintain a safe gap between requests

    # 4. Merge data and flatten structure
    final_corpus = []
    
    # Group sentences by lemma for CSV flattening
    grouped_data = {}
    for item in all_sentences:
        if item['lemma'] not in grouped_data:
            grouped_data[item['lemma']] = []
        grouped_data[item['lemma']].append(item)

    for word, sentences in grouped_data.items():
        row = {
            "Lemma": word, 
            "Source": sentences[0]['Source'], 
            "CEFR_Target": sentences[0]['CEFR_Target']
        }
        
        # Ensure only 3 sentences are processed (HIGH, LOW, NEUTRAL)
        for i, item in enumerate(sentences[:3]):
            key_prefix = f"Sentence_{i+1}_{item['register']}"
            sentence_text = item['sentence']
            
            row[f"{key_prefix}_Text"] = sentence_text
            row[f"{key_prefix}_Mood"] = item['mood']
            row[f"{key_prefix}_POS"] = pos_lookup.get(sentence_text, "N/A")
        
        final_corpus.append(row)

    # 5. Save to CSV
    if final_corpus:
        df = pd.DataFrame(final_corpus)
        
        # Define the desired column order
        new_columns = ["Lemma", "Source", "CEFR_Target"]
        for i in range(1, 4):
            for reg in ["HIGH", "LOW", "NEUTRAL"]:
                new_columns.extend([
                    f"Sentence_{i}_{reg}_Text", 
                    f"Sentence_{i}_{reg}_Mood", 
                    f"Sentence_{i}_{reg}_POS"
                ])
        
        # Select and reorder columns
        df = df[[col for col in new_columns if col in df.columns]]
        
        df.to_csv(OUTPUT_FILENAME, index=False)
        print(f"\n SUCCESS! Final analyzed data saved to '{OUTPUT_FILENAME}'.")
        print(f"Corpus size: {len(df)} entries.")
        print("--- Generated Data Head ---")
        print(df.head())
    else:
        print("\n Failed to generate the final CSV.")

if __name__ == "__main__":
    main()