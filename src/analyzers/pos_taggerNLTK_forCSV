import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# --- CONFIGURATION ---
INPUT_FILENAME = "outputs/thesis_semantic_data_final.csv" # Update path if needed
OUTPUT_FILENAME = "thesis_textbook_data_filtered.csv"      # Update path if needed

# Download required NLTK datasets (quietly)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()

def main():
    print("--- STARTING NLTK POS-GATEKEEPER (CSV VERSION) ---")
    
    try:
        # Load the CSV
        df = pd.read_csv(INPUT_FILENAME)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return
        
    print(f"Total raw sentences loaded: {len(df)}")
    
    valid_indices = []
    
    # Identify the correct column names (handling uppercase/lowercase differences)
    text_col = 'Full_Sentence' if 'Full_Sentence' in df.columns else 'sentence'
    lemma_col = 'Lemma' if 'Lemma' in df.columns else 'lemma'
    
    # 1. The Lemmatize-Before-Filtering Logic
    for index, row in df.iterrows():
        target_lemma = str(row[lemma_col]).lower().strip()
        sentence_text = str(row[text_col])
        
        # Tokenize and tag the sentence
        tokens = word_tokenize(sentence_text)
        pos_tags = nltk.pos_tag(tokens)
        
        is_valid_verb = False
        
        # Scan the sentence for our target verb
        for token, tag in pos_tags:
            # Check if NLTK tagged it as ANY type of verb (VB, VBD, VBG, VBN, VBP, VBZ)
            if tag.startswith('V'):
                # Lemmatise the verb
                token_lemma = lemmatizer.lemmatize(token.lower(), pos='v')
                
                # Check if it matches our target lemma
                if token_lemma == target_lemma:
                    is_valid_verb = True
                    break # Found it acting as a verb, keep the row
                    
        if is_valid_verb:
            valid_indices.append(index)

    # 2. Filter the dataframe to only keep valid rows
    clean_df = df.loc[valid_indices]
    dropped_count = len(df) - len(clean_df)

    print(f"\nFiltering Complete!")
    print(f"Sentences Kept: {len(clean_df)}")
    print(f"Sentences Dropped (Nouns/Adjectives): {dropped_count}")

    # 3. Export the clean data
    if not clean_df.empty:
        clean_df.to_csv(OUTPUT_FILENAME, index=False)
        print(f"Clean, verb-only dataset saved to '{OUTPUT_FILENAME}'.")
    else:
        print("\nFailed to generate the final CSV. No valid verbs found.")

if __name__ == "__main__":
    main()