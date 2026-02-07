import os
import google.generativeai as genai
import time
import json
import re

# --- CONFIGURATION ---
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise RuntimeError("Missing GOOGLE_API_KEY environment variable.")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

# The words to analyze
TARGET_WORDS = ["run", "set", "break", "draw", "mean", "hold", "take", "turn"] 

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
    cleaned_text = re.sub(r'```json|```', '', raw_text, flags=re.IGNORECASE).strip()
    start_index = cleaned_text.find('[')
    if start_index != -1:
        cleaned_text = cleaned_text[start_index:]
        
    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        print(f"Raw text causing error (truncated): {cleaned_text[:200]}...")
        return None

def generate_pragmatic_data(word):
    """Generates the three required sentences for a single word."""
    print(f"Generating pragmatic data for: '{word}'...")
    
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
        print(f"Error processing pragmatic data for '{word}': {e}")
        return None

def main():
    all_sentences = []
    print("--- STARTING PRAGMATIC SENTENCE GENERATION ---")
    
    for word in TARGET_WORDS:
        pragmatic_data = generate_pragmatic_data(word)
        
        if pragmatic_data:
            all_sentences.extend(pragmatic_data)
        
        time.sleep(2) # Be nice to the API
        
    if all_sentences:
        filename = "outputs/intermediate_sentences.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(all_sentences, f, indent=4)
            
        print(f"\n SUCCESS! {len(all_sentences)} sentences generated and saved to '{filename}'.")
        print("NEXT STEP: Run 'pos_tagger.py' to add POS tags and create the final CSV.")
    else:
        print("\n Failed to generate any data. Check API Key and network connection.")

if __name__ == "__main__":
    main()