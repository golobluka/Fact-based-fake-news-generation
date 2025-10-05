import re, json
import sys, os
import ollama

#Get prompts from prompt package
prompt_package_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
sys.path.append(prompt_package_directory)
from prompts.fact_extraction import GENERAL_TOPICS, GENERAL_CHANGES_CHANGE_MEANING


def find_json(text, generalized_topics=GENERAL_TOPICS):
    """Function to find JSON-like content in the text."""
    content = []

    # Dynamically construct pattern0, pattern1, and pattern2 using generalized topics
    def build_pattern(topic_list, delimiter=r',.*\n', closing_pattern=r'\s*\n', opening_delimiter = ''):
        pattern = ""
        for topic in topic_list:
            key = topic['Name of fact']
            # Use re.escape to escape special characters in the key
            pattern += r'.*' + re.escape(key) + r'.*:' + opening_delimiter + '\s*(.*)' + delimiter
        # Ensure the closing pattern has balanced parentheses
        pattern = pattern.rstrip(delimiter) + closing_pattern

        # Debugging: Print the generated pattern
        print("Generated pattern:", pattern)

        return pattern

    def build_pattern_second(topics):
        """
        Generate a regular expression pattern based on a list of topics.

        Parameters:
        topics (list): List of topic names as strings.

        Returns:
        str: A regular expression string.
        """
        pattern = r'(?:'

        for i, topic in enumerate(topics):
            name = topic["Name of fact"]
            # Escape special characters in topic names (like spaces)
            escaped_topic = re.escape(name)
            # Append the part of the regex for this topic
            pattern += f'"{escaped_topic}": \\[([^\\]]*?)\\]'
            # Add comma and newline characters between topics, but not after the last one
            if i < len(topics) - 1:
                pattern += r',?\n\s*'

        # Close the non-capturing group
        pattern += r')'

        return pattern

    # Build patterns based on generalized topics
    pattern0 = build_pattern(generalized_topics)
    pattern1 = build_pattern(generalized_topics, delimiter=r',.*', closing_pattern=r'\}.*')
    pattern2 = build_pattern_second(generalized_topics)

    patterns = [pattern2, pattern0, pattern1]

    # Refactored matching logic to avoid code duplication
    for pattern in patterns:
        try:
            matches = re.finditer(pattern, text)
            for match in matches:
                content_dict = {}
                for i, topic in enumerate(generalized_topics, start=1):
                    # Ensure that match groups are not None before replacing quotes
                    matched_value = match.group(i)
                    if matched_value:
                        content_dict[topic["Name of fact"]] = matched_value.replace('\"', '')
                content.append(content_dict)
        except re.error as e:
            # Log or print the regex error for debugging
            print(f"Regex error with pattern: {pattern}. Error: {e}")

    return content

def extract_last_article(text):
    """Function to extract the last article."""
    pattern = r"BEGINNING OF THE ARTICLE(.*?)END OF THE ARTICLE"
    matches = list(re.finditer(pattern, text, re.DOTALL))
    if matches:
        last_match = matches[-1]
        return last_match.group(1).strip()
    return None

def extract_last_text(text):
    """Function to extract the last text."""
    pattern = r"BEGINNING OF THE TEXT(.*?)END OF THE TEXT"
    matches = list(re.finditer(pattern, text, re.DOTALL))
    if matches:
        last_match = matches[-1]
        return last_match.group(1).strip()
    return None

def print_readable_dict(data):
    """Prints a dictionary in a readable JSON-like format."""
    print(json.dumps(data, indent=4, ensure_ascii=False))

def extract_fact_from_text(text):
    pattern1 = r"BEGINNING OF FACTS(.*?)END OF FACTS"
    pattern2 = r"BEGINING OF FACTS(.*?)END OF FACTS" #I don't know why but sometimes BEGINNING is written with one N.
    match = re.search(pattern1, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        match2 = re.search(pattern2, text, re.DOTALL)
        if match2:
            return match2.group(1).strip()
    return None

def generate_aggressive_prompts(topics):
    changes_aggressive = []

    # Iterate over each topic to construct the changes_aggressive entry
    for topic in topics:
        name_of_fact = topic['Name of fact']
        description_of_fact = topic['Description of fact'].strip()
        common_examples = topic["Common examples"]
        
        # Construct the detailed instruction
        changing_orders = GENERAL_CHANGES_CHANGE_MEANING.format(name_of_fact=name_of_fact, description_of_fact=description_of_fact, common_examples=common_examples)
        
        changes_aggressive.append({"Name of fact": name_of_fact, "Changing orders": changing_orders})

    return changes_aggressive


def model_response(model_name, prompt, prompt_variables = {}):
    """ takes ollama model and generates response relative to the given prompt."""
    formatted_content = prompt.format(**prompt_variables)
    response = ollama.chat(model=model_name, messages=[
        {
            'role': 'user',
            'content': formatted_content,
            'temperature': 0.2,
        }
    ])
    return response['message']['content']