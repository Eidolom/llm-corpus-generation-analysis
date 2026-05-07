import google.generativeai as genai

# Paste key directly here for the test
API_KEY = "API_KEY_HERE"

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

print("--- TESTING GEMINI API ---")

try:
    response = model.generate_content("Write a sentence using the word 'run'.")
    print(f"Success! Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")