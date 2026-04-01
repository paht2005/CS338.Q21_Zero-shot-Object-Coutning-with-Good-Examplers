"""
Prompt Enhancement using Gemini API
Enhanced prompts improve detection and counting accuracy
"""

import google.generativeai as genai
from PIL import Image
import inflect
import time

# Initialize
p = inflect.engine()

# Configure Gemini (user should set their own API key)
GEMINI_API_KEY = "AIzaSyD4JRCmtzblaw33zzHvKq01Xbg_kshlM5c"  # Replace with your key
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash-exp')


def enhance_prompt_with_gemini(image, class_name, max_retries=2):
    """
    Enhance prompt using Gemini Vision API
    
    Args:
        image: PIL Image
        class_name: Base class name (e.g., "dog", "car")
        max_retries: Number of retry attempts
    
    Returns:
        Enhanced prompt string or original class_name if fails
    """
    # 1. Singularize
    singular_name = p.singular_noun(class_name)
    if not singular_name:
        singular_name = class_name
    
    # 2. Craft prompt for "Visual Definition"
    prompt = f"""
Look at the image and provide the **visual definition** of a single '{singular_name}'.

**CRITICAL RULES:**
1. **IF the image does NOT contain any '{singular_name}'**, respond with ONLY: "single {singular_name} ."
2. **IF the image DOES contain '{singular_name}'**, describe ONE instance's visual appearance.

**Task**: Describe the intrinsic physical appearance of just **ONE** instance, as if it were cropped out and isolated.

**Format Rules:**
- Start with 'single {singular_name}'
- Use dot-separated phrases (e.g., "single dog . brown fur . four legs .")
- Focus on: Shape, Color, Material, Texture
- Ignore background or other objects
- End with a dot

**Example for 'keyboard key':** 
BAD: keyboard key . rows of buttons . full keyboard layout .
GOOD: single keyboard key . square shape . black plastic material . white printed letter . smooth surface .

**Your output for '{singular_name}':**
"""
    
    # 3. Call Gemini API with retry
    for attempt in range(max_retries):
        try:
            response = model.generate_content([prompt, image])
            text = response.text.strip().replace("\n", " ").replace("..", ".")
            
            # Ensure ends with dot
            if not text.endswith('.'):
                text += ' .'
            
            # Validate response
            if text and len(text) > 5:
                return text
            else:
                # Empty/invalid response, return fallback
                return f"single {singular_name} ."
                
        except Exception as e:
            print(f"⚠ Gemini API error (attempt {attempt+1}/{max_retries}): {e}")
            
            # Handle rate limit
            if "429" in str(e) or "quota" in str(e).lower():
                if attempt < max_retries - 1:
                    wait_time = 5 * (attempt + 1)
                    print(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
            
            # Handle "object not found" gracefully
            if attempt == max_retries - 1:
                return f"single {singular_name} ."
    
    # Fallback
    return f"single {singular_name} ."


def enhance_prompt_simple(class_name):
    """
    Simple prompt enhancement without Gemini (fallback)
    Just singularizes and formats
    """
    singular_name = p.singular_noun(class_name)
    if not singular_name:
        singular_name = class_name
    
    return f"single {singular_name} ."


# Test function
if __name__ == "__main__":
    # Test with sample image
    test_image_path = "./data/FSC147/images_384_VarV2/2.jpg"
    test_class = "dog"
    
    try:
        img = Image.open(test_image_path)
        enhanced = enhance_prompt_with_gemini(img, test_class)
        print(f"Original: {test_class}")
        print(f"Enhanced: {enhanced}")
    except Exception as e:
        print(f"Test failed: {e}")
