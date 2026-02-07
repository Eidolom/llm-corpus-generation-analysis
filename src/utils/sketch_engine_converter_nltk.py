import json
import re
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag

# --- NLTK SETUP ---
# NLTK requires downloading model files (tokenizers, taggers). 
try:
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    print("⬇️ Downloading necessary NLTK models...")
    nltk.download('punkt_tab')
    nltk.download('averaged_perceptron_tagger_eng')

# --- CONFIGURATION ---
INPUT_FILE = 'data/textbook_sentences.json'
OUTPUT_JSON = 'data/semantic_analysis_results.json'
OUTPUT_CSV = 'data/semantic_analysis_summary.csv'

def clean_text(text):
    """Cleans messy textbook artifacts."""
    if not text: return ""
    # Remove double single quotes often found in SQL dumps
    text = text.replace("''", "'")
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_context_window(words, index, window_size=5):
    """
    Since NLTK doesn't do full dependency parsing easily, 
    we grab a 'context window' of words following the verb 
    to capture the object/phrase (e.g., 'keep' -> 'in touch').
    """
    start = index + 1
    end = min(index + 1 + window_size, len(words))
    context_words = words[start:end]
    return " ".join(context_words)

def analyze_data(data):
    analyzed_rows = []
    print(f"Processing {len(data)} raw entries with NLTK...")

    for entry in data:
        raw_text = clean_text(entry.get('sentence', ''))
        target_lemma = entry.get('lemma', '').lower()
        
        # 1. Split chunk into sentences
        sentences = sent_tokenize(raw_text)
        
        for sent in sentences:
            # 2. Tokenize words
            words = word_tokenize(sent)
            
            # 3. Part-of-Speech Tagging
            # Returns list of tuples: [('Word', 'TAG'), ...]
            tagged_words = pos_tag(words)
            
            found_verb = False
            context_pattern = "N/A"
            
            for i, (word, tag) in enumerate(tagged_words):
                # Check if word matches target (normalized)
                # AND if tag starts with 'V' (VB, VBD, VBG, VBN, VBP, VBZ are all verbs)
                if word.lower() == target_lemma and tag.startswith('V'):
                    found_verb = True
                    # Grab the words following the verb
                    context_pattern = get_context_window(words, i)
                    break 
            
            # 4. Filter and Save
            # We filter out very short fragments (e.g. headers)
            if found_verb and len(words) > 3:
                row = {
                    "Target_Lemma": target_lemma,
                    "Extracted_Sentence": sent,
                    "Context_Pattern": f"{target_lemma} + {context_pattern}",
                    "Original_Source": entry.get('Source', 'Unknown'),
                    "Register": entry.get('register', 'N/A')
                }
                analyzed_rows.append(row)

    return analyzed_rows

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("--- STARTING SEMANTIC ANALYSIS (NLTK VERSION) ---")
    
    # 1. Load Data
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find '{INPUT_FILE}'. Make sure it exists.")
        exit()
        
    # 2. Analyze
    results = analyze_data(raw_data)
    
    # 3. Save Results
    if results:
        with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
            
        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_CSV, index=False)
        
        print(f"   Analysis Complete.")
        print(f"   Processed {len(raw_data)} chunks into {len(results)} specific sentences.")
        print(f"   Saved to '{OUTPUT_CSV}'.")
    else:
        print("No valid sentences containing the target verbs were found.")