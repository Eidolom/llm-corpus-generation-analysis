import os
import google.generativeai as genai

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("Missing GOOGLE_API_KEY environment variable.")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

print("--- TESTING GEMINI API ---")

try:
    response = model.generate_content("Write a sentence using the word 'run'.")
    print(f"Success! Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")