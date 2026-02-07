import os
import google.generativeai as genai
import time
import json
import re

# --- CONFIGURATION ---
API_KEY = os.getenv("GOOGLE_API_KEY")
TARGET_WORDS_FILE = "data/target_words.txt"
OUTPUT_FILENAME = "outputs/intermediate_sentences.json"

if not API_KEY:
    raise RuntimeError("Missing GOOGLE_API_KEY environment variable.")

genai.configure(api_key=API_KEY)
# Using gemini-2.5-flash for its speed and cost-effectiveness for high-volume data generation
model = genai.GenerativeModel('gemini-2.5-flash')

# The "Pragmatics" Prompt
SYSTEM_PROMPT_PRAGMATICS = """
You are a linguistic expert specializing in pragmatics.
Generate THREE distinct example sentences for the target word. 
Each sentence must strictly satisfy the corresponding register, mood, and context requirements.
The sentences must be appropriate for a CEFR B1 learner.

Output ONLY a strict JSON array of objects. Do not include any explanations or markdown.
"""

def clean_and_load_json(raw_text):
    """Helper function to clean markdown and load JSON safely."""
    if not raw_text:
        return None
    # Remove any markdown formatting (e.g., ```json)
    cleaned_text = re.sub(r'```json|```', '', raw_text, flags=re.IGNORECASE).strip()
    
    # Attempt to find the start of the JSON array
    start_index = cleaned_text.find('[')
    if start_index != -1:
        cleaned_text = cleaned_text[start_index:]
        
    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error (Truncated Text: {cleaned_text[:100]}...): {e}")
        return None

def load_target_words(filename):
    """Loads a list of words from a file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            # Filter out empty lines and strip whitespace
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f" Error: Target words file '{filename}' not found. Please create it.")
        return None

def generate_pragmatic_data(word):
    """Generates the three required sentences for a single word."""
    
    prompt = f"""
    Target Word: {word}
    
    Generate 3 sentences with these characteristics:
    1. Register: HIGH (Formal/Polite), Mood: Question, Context: Workplace/Request.
    2. Register: LOW (Informal/Casual), Mood: Statement, Context: Friends/Weekend Plans.
    3. Register: NEUTRAL, Mood: Imperative/Command, Context: Directions/Instructions.
    
    Output the result as a JSON array with keys: 'register', 'mood', 'sentence'.
    """
    
    try:
        response = model.generate_content(
            contents=[{"role": "user", "parts": [{"text": SYSTEM_PROMPT_PRAGMATICS + "\n\n" + prompt}]}]
        )
        data = clean_and_load_json(response.text)
        
        # Add the fixed metadata to each item
        if data:
            for item in data:
                item['lemma'] = word
                item['Source'] = "Synthetic (Gemini 2.5)"
                item['CEFR_Target'] = "B1"
                
        return data
        
    except Exception as e:
        print(f" Error processing pragmatic data for '{word}': {e}")
        return None

def main():
    if not API_KEY:
        print(" ERROR: Please set GOOGLE_API_KEY in your environment.")
        return
        
    TARGET_WORDS = load_target_words(TARGET_WORDS_FILE)
    if not TARGET_WORDS:
        return

    all_sentences = []
    total_words = len(TARGET_WORDS)
    success_count = 0
    
    print(f"--- STARTING PRAGMATIC SENTENCE GENERATION FOR {total_words} WORDS ---")
    
    for i, word in enumerate(TARGET_WORDS):
        print(f"[{i+1}/{total_words}]  Generating data for: '{word}'...")
        pragmatic_data = generate_pragmatic_data(word)
        
        if pragmatic_data:
            all_sentences.extend(pragmatic_data)
            success_count += 1
        
        time.sleep(1.5) # Maintain a safe gap between requests
        
    if all_sentences:
        with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
            json.dump(all_sentences, f, indent=4)
            
        print(f"\n SUCCESS! {len(all_sentences)} sentences generated ({success_count}/{total_words} words successful).")
        print(f"Data saved to '{OUTPUT_FILENAME}'.")
        print("NEXT STEP: Run 'pos_tagger.py' to add POS tags and create the final CSV.")
    else:
        print("\n Failed to generate any data. Check API Key and prompt quality.")

if __name__ == "__main__":
    main()