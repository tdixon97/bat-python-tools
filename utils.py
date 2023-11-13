import re



def find_and_capture(text:str, pattern:str):
    """Function to  find the start index of a pattern in a string
    Parameters:
        - text: string
        - pattern: string
    Returns:
        -index: Integer corresponding to the first letter of the first instance of the pattern
    """
    match = re.search(pattern, text)
    
    if match:
        start_index = match.end()
        return start_index
    else:
        raise ValueError("pattern {} not found in text {} ".format(pattern,text))
    
