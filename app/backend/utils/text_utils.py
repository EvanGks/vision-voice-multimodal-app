import re

def clean_text_for_tts(text):
    """
    Clean markdown and LaTeX formatting from text to make it suitable for TTS.
    
    Args:
        text: Text with markdown/LaTeX formatting
        
    Returns:
        Clean text suitable for TTS
    """
    # Handle math expressions in LaTeX format
    # Replace inline math $...$ with "the expression ..."
    def replace_inline_math(match):
        expr = match.group(1).strip()
        # Basic LaTeX cleaning - replace common math symbols with spoken equivalents
        expr = expr.replace('\\times', ' times ')
        expr = expr.replace('\\div', ' divided by ')
        expr = expr.replace('\\cdot', ' dot ')
        expr = expr.replace('\\frac', ' fraction ')
        expr = expr.replace('{', ' ').replace('}', ' ')
        expr = expr.replace('\\sqrt', ' square root of ')
        expr = expr.replace('^', ' to the power of ')
        expr = expr.replace('_', ' sub ')
        expr = expr.replace('\\pi', ' pi ')
        expr = expr.replace('\\infty', ' infinity ')
        expr = expr.replace('\\sum', ' sum ')
        expr = expr.replace('\\int', ' integral ')
        expr = re.sub(r'\\[a-zA-Z]+', ' ', expr)  # Remove other LaTeX commands
        return f" {expr} "
    
    # Replace block math $$...$$ with "the expression ..."
    text = re.sub(r'\$\$(.*?)\$\$', replace_inline_math, text, flags=re.DOTALL)
    text = re.sub(r'\$(.*?)\$', replace_inline_math, text)
    
    # Replace markdown headers with plain text and emphasis
    text = re.sub(r'^#{1,6}\s+(.*?)$', r'\1.', text, flags=re.MULTILINE)
    
    # Replace markdown bold/italic with plain text
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
    text = re.sub(r'\*(.*?)\*', r'\1', text)      # Italic
    text = re.sub(r'__(.*?)__', r'\1', text)      # Bold
    text = re.sub(r'_(.*?)_', r'\1', text)        # Italic
    
    # Replace markdown lists with plain text
    text = re.sub(r'^\s*[\*\-\+]\s+(.*?)$', r'• \1', text, flags=re.MULTILINE)  # Unordered lists
    text = re.sub(r'^\s*\d+\.\s+(.*?)$', r'\1', text, flags=re.MULTILINE)      # Ordered lists
    
    # Replace markdown code blocks with plain text
    text = re.sub(r'```(?:.*?)\n(.*?)```', r'\1', text, flags=re.DOTALL)  # Code blocks
    text = re.sub(r'`(.*?)`', r'\1', text)  # Inline code
    
    # Replace markdown links with just the text
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    
    # Replace markdown images with alt text or placeholder
    text = re.sub(r'!\[(.*?)\]\(.*?\)', r'Image: \1', text)
    
    # Replace horizontal rules
    text = re.sub(r'^-{3,}$', ' ', text, flags=re.MULTILINE)
    
    # Replace multiple newlines with a single one
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Replace markdown quotes with plain text
    text = re.sub(r'^\s*>\s+(.*?)$', r'\1', text, flags=re.MULTILINE)
    
    # Fix any remaining markdown artifacts
    text = text.replace('\\', '')
    text = text.replace('**', '')
    text = text.replace('*', '')
    text = text.replace('__', '')
    text = text.replace('_', '')
    text = text.replace('#', '')
    text = text.replace('```', '')
    text = text.replace('`', '')
    
    return text 