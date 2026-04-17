import sys
import json
import pandas as pd
from pathlib import Path
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

INPUT_FILENAME = "outputs/intermediate_sentences.json"
OUTPUT_FILENAME = "outputs/thesis_pragmatic_data_filtered.csv"

nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()

def load_sentences(filename):
    """Loads sentences from the intermediate JSON file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def main():
    print("--- STARTING NLTK POS-GATEKEEPER ---")
    all_sentences = load_sentences(INPUT_FILENAME)
    if not all_sentences:
        return

    print(f"Total raw sentences loaded: {len(all_sentences)}")
    
    valid_sentences = []
    dropped_count = 0

    for item in all_sentences:
        target_lemma = item['lemma'].lower()
        sentence_text = item['sentence']
        
        tokens = word_tokenize(sentence_text)
        pos_tags = nltk.pos_tag(tokens)
        
        is_valid_verb = False
        
        for token, tag in pos_tags:
            if tag.startswith('V'):
                token_lemma = lemmatizer.lemmatize(token.lower(), pos='v')
                
                if token_lemma == target_lemma:
                    is_valid_verb = True
                    break
                    
        if is_valid_verb:
            valid_sentences.append(item)
        else:
            dropped_count += 1

    print(f"\nFiltering Complete!")
    print(f"Sentences Kept: {len(valid_sentences)}")
    print(f"Sentences Dropped (Nouns/Adjectives): {dropped_count}")

    if valid_sentences:
        df = pd.DataFrame(valid_sentences)
        df.to_csv(OUTPUT_FILENAME, index=False)
        print(f"Clean, verb-only dataset saved to '{OUTPUT_FILENAME}'.")
        
    else:
        print("\nFailed to generate the final CSV. No valid verbs found.")

if __name__ == "__main__":
    main()