from textblob import TextBlob

def correct_spelling(text):
    """
    Corrects the spelling of input text using TextBlob.
    
    Parameters:
    text (str): Input text to be corrected.
    
    Returns:
    str: Corrected text.
    """
    blob = TextBlob(text)
    corrected_text = blob.correct()
    return str(corrected_text)

# Example usage:
input_text = "I hdd a pen."
corrected_text = correct_spelling(input_text)
print("Input Text:", input_text)
print("Corrected Text:", corrected_text)
