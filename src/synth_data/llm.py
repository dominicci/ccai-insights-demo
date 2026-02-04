# src/synth_data/llm.py

import os
import json
from tenacity import retry, stop_after_attempt, wait_exponential, wait_random

# Define retry Logic for internal calls
retry_decorator = retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60) + wait_random(min=0, max=5),
    reraise=True
)

def get_llm_response(prompt: str) -> str:
    """
    Tries to get a response from Gemini (Google) first, then OpenAI.
    Returns the raw string content.
    """
    
    # 1. Try Google Gemini
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    if google_api_key:
        try:
            import google.generativeai as genai
            
            @retry_decorator
            def call_gemini():
                genai.configure(api_key=google_api_key)
                model = genai.GenerativeModel("gemini-2.5-pro")
                response = model.generate_content(prompt)
                return response.text

            print(" Using Google Gemini...")
            return call_gemini()
            
        except ImportError:
            print("  [Warning] google-generativeai not installed. Skipping Gemini.")
        except Exception as e:
            # Fallback to OpenAI if Gemini fails
            print(f"  [Error] Gemini generation exhausted or failed: {e}")

    # 2. Try OpenAI
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if openai_api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_api_key)
            
            @retry_decorator
            def call_openai():
                completion = client.chat.completions.create(
                    model="gpt-4o-mini",  # or gpt-3.5-turbo
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that generates synthetic data in JSON format."},
                        {"role": "user", "content": prompt}
                    ]
                )
                return completion.choices[0].message.content

            print(" Using OpenAI...")
            return call_openai()
            
        except ImportError:
            print("  [Warning] openai library not installed. Skipping OpenAI.")
        except Exception as e:
            print(f"  [Error] OpenAI generation exhausted or failed: {e}")

    raise EnvironmentError("No valid API key found (GOOGLE_API_KEY or OPENAI_API_KEY) or libraries missing.")
