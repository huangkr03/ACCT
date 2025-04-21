

def check_loop(text, check_repeat=100):
	"""
	Check whether the output of the LLM (text) suffers from infinite loop problem
	by detecting repetitive patterns in the text.
	
	Args:
		text (str): The text output from an LLM
	
	Returns:
		bool: True if an infinite loop is detected, False otherwise
	"""
    