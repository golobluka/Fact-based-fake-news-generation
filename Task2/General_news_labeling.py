


#Imports
import random
import ollama
import json
import re
import os
import pandas as pd

import sys
current_path = os.getcwd()  # Get the current working directory
parent_directory = os.path.dirname(current_path)
# Add the directory you want to import from
sys.path.append(parent_directory)


# Attempted Import
try:
    from constants import (
        GENERAL_NAME_OF_FACT, 
        GENERAL_DESCRIPTION_OF_FACT, 
        GENERAL_COMMON_EXAMPLES,
        testing_dictionary
    )
    print("Imported Constants Successfully.")
except ImportError as e:
    print("ImportError:", e)
    # Print available attributes from the constants module
    import constants
    print("Available Constants:", dir(constants))
    raise  # Raise the error after printing







#________________________________________________________________
# prompts

GENERAL_TOPICS = [
    {
        GENERAL_NAME_OF_FACT: 'Name of casualty or group', 
        GENERAL_DESCRIPTION_OF_FACT: ' represents the casualties names or the names of the groups associated with the casualties.', 
        GENERAL_COMMON_EXAMPLES: 'men, soldiers, children'
    },
    {
        GENERAL_NAME_OF_FACT: 'Gender or age group', 
        GENERAL_DESCRIPTION_OF_FACT: ' of casualty indicates if the casualties are male or female, or specify their age group .', 
        GENERAL_COMMON_EXAMPLES: 'Male, Female, Child, Adult, Senior'
    },
    {
        GENERAL_NAME_OF_FACT: 'Cause of death', 
        GENERAL_DESCRIPTION_OF_FACT: ' specifies the weapons used by the aggressor (e.g., shooting, shelling, chemical weapons, etc.)', 
        GENERAL_COMMON_EXAMPLES: 'Shooting, Shelling, Chemical weapons'
    },
    {
        GENERAL_NAME_OF_FACT: 'Type', 
        GENERAL_DESCRIPTION_OF_FACT: ' of casualty classifies the casualties as a civilian or non-civilian (e.g., military personnel are non-civilians).', 
        GENERAL_COMMON_EXAMPLES: 'Civilian, Non-civilian'
    },
    {
        GENERAL_NAME_OF_FACT: 'Actor', 
        GENERAL_DESCRIPTION_OF_FACT: ' identifies the actors responsible for the incident, such as rebel groups, Russian forces, ISIS, the Syrian army, U.S. military, etc.', 
        GENERAL_COMMON_EXAMPLES: 'Rebel groups, Russian forces, ISIS'
    },
    {
        GENERAL_NAME_OF_FACT: 'Place of death', 
        GENERAL_DESCRIPTION_OF_FACT: ' specifies the locations where the attacks occurred (e.g., Aleppo, Damascus, Homs, Idlib, Raqqa, Daraa, Deir ez-Zor, Qamishli, Palmyra, etc.).', 
        GENERAL_COMMON_EXAMPLES: 'Aleppo, Damascus, Homs'
    },
    {
        GENERAL_NAME_OF_FACT: 'Date of death', 
        GENERAL_DESCRIPTION_OF_FACT: ' provides the dates when the attacks occurred.', 
        GENERAL_COMMON_EXAMPLES: '2021-01-01, 2022-06-15'
    }
]


# Constant for shape of question
GENERAL_SHAPE_OF_QUESTION = """The \"{name_of_fact}\" is defined as: {description_of_fact}. 

Is the \"{name_of_fact}\" in the article approximately coherent with this description: {{}}? All content in description must be contained in the article and all information about \"{name_of_fact}\" must mentioned in description. Describe your thinking procedure and output "The answer is true" or "The answer is false". """

GENERAL_SHAPE_OF_QUESTION_2 = """{description_of_fact}. 

will provide you with two pieces of information about "\{name_of_fact}\" 
    1. {{}}
    2. {{}}
Do these two pieces of information convey the same meaning? Please describe your reasoning process and conclude with either "The answer is true" or "The answer is false."""

# Prompts
GENERAL_PROMPT = """Please read and understand the event that is stored in JSON format:

{events}

You must check that the event presented in the article is from among previously red events. Try to check that all the information matches. That means that "Name of casualty or group" "Gender or age group", "Cause of death", "Type", "Actor", "Place of death" and "Date of death" must match.

{article}

If the article matches some event print 'true', else print 'false'. In addition to 'true' or 'false' provide explanation.
"""

GENERAL_PROMPT_ONE_BY_ONE = """Carefully read through the article and try to understand its {topic}. {meaning_of_topic}

BEGINNING OF TEXT
{article}
END OF TEXT

{question}
"""









#________________________________________________________________
#Suplementary functions


def find_json(text, general_topics = GENERAL_TOPICS):
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
    pattern0 = build_pattern(general_topics)
    pattern1 = build_pattern(general_topics, delimiter=r',.*', closing_pattern=r'\}.*')
    pattern2 = build_pattern_second(general_topics)

    # List of patterns to apply
    patterns = [pattern2, pattern0, pattern1]

    # Refactored matching logic to avoid code duplication
    for pattern in patterns:
        try:
            matches = re.finditer(pattern, text)
            for match in matches:
                content_dict = {}
                for i, topic in enumerate(general_topics, start=1):
                    # Ensure that match groups are not None before replacing quotes
                    matched_value = match.group(i)
                    if matched_value:
                        content_dict[topic["Name of fact"]] = matched_value.replace('\"', '')
                content.append(content_dict)
        except re.error as e:
            # Log or print the regex error for debugging
            print(f"Regex error with pattern: {pattern}. Error: {e}")

    return content

def find_first_true_or_false(text):
    """Finds the first occurrence of the words 'true' or 'false' in the text and returns it with its position."""
    # Regular expression pattern to match 'true' or 'false'
    pattern = r'\b(The answer is true|The answer is false|The answer is True|The answer is False|The answer is TRUE|The answer is FALSE|the answer is true|the answer is false|the answer is True|the answer is False|the answer is TRUE|the answer is FALSE|The answer is \*\*False\*\*|The answer is \*\*True\*\*|The answer is \*\*false\*\*|The answer is \*\*true\*\*|the answer is \*\*False\*\*|the answer is \*\*True\*\*|the answer is \*\*false\*\*|the answer is \*\*true\*\*)\b'

    # Search for the first occurrence
    match = re.search(pattern, text)  # Using IGNORECASE to match 'True', 'False' etc.

    if match:
        word = match.group(1)  # Get the matched word ('true' or 'false')
        return word.lower()
    else:
        return None # Return None if not found, and -1 for position
    
def generate_one_by_one_prompts(topics):
    questions = []
    for topic in topics:
        question_for_fact = GENERAL_SHAPE_OF_QUESTION.format(name_of_fact = topic["Name of fact"], description_of_fact = topic["Description of fact"], common_examples=topic["Common examples"])
        dictionary = {"Name of fact": topic["Name of fact"], "Description of fact": topic["Description of fact"], "Question for fact": question_for_fact}
        questions.append(dictionary)
    return questions

def generate_one_by_one_prompts_2(topics):
    questions = []
    for topic in topics:
        question_for_fact = GENERAL_SHAPE_OF_QUESTION_2.format(name_of_fact = topic["Name of fact"], description_of_fact = topic["Description of fact"], common_examples=topic["Common examples"])
        dictionary = {"Name of fact": topic["Name of fact"], "Description of fact": topic["Description of fact"], "Question for fact": question_for_fact}
        questions.append(dictionary)
    return questions


def model_response(model_name, prompt, prompt_variables):
    if prompt_variables != None:
        formatted_content = prompt.format(**prompt_variables)
    else:
        formatted_content = prompt
    response = ollama.chat(model=model_name, messages=[
        {
            'role': 'user',
            'content': formatted_content,
            'temperature': 0.2,
        }
    ])
    return response['message']['content']

def transform_list_to_dataframe(data_list):
    # Define the column metadata
    columns = [
        "Labeled correctly in first case", 
        "Successful labelings in first case", 
        "Labeled correctly in second case", 
        "Successful labelings in second case"
    ]
    
    # Convert list to pandas DataFrame
    df = pd.DataFrame(data_list, columns=columns)
    
    return df















#______________________________________________________________________________________-
#main functions


def fake_detect_comparison_true_to_true_and_false_to_true(list_of_changed_articles, general_topics, model_name = "llama3.1:8b", print_comments = False, testing = False): 
    """
Inputs:
    list_of_changed_articles: A list containing pairs of true and false articles along with their facts and changes. The format of the list is as follows:
    [
    ["Index", "True Article Content", "Facts table", "Fake article content", "Changed topics", "Facts", "Changed Topic 1", "Changed Topic 2"]
    ]
    model_name: The name of the model to be used for comparison (default is "llama3.1:8b").
    print_comments: A boolean flag to print intermediate comments (default is False).

Flow:
    - Initialize an empty list 'data' to store results.
    - Iterate over each article pair in list_of_changed_articles.
    - For each true article, generate responses for each topic using the model and count the number of true and false responses.
    - Repeat the process for the corresponding false article.
    - Append the results to 'data' and return it as a pandas DataFrame.

Outputs:
    Returns a pandas DataFrame containing the counts of true and false responses for each article pair. The DataFrame will look like this:
    
    Example:
    |   False Negatives |   Computed Outputs (True Article) |   True Negatives |   Computed Outputs (False Article) |
    |-------------------|-----------------------------------|------------------|------------------------------------|
    |                 0 |                                 7 |                2 |                                  7 |
    |                 1 |                                 7 |                3 |                                  7 |
    
    - "False Negatives": number of false negatives (true article is compared to true facts, so ideally this should sum to 0).
    - "Computed Outputs (True Article)": number of computed outputs for true article comparison. This should sum up to all cases (in this case, 7).
    - "True Negatives": number of true negatives (false article is compared to true facts, so ideally this should sum to 2).
    - "Computed Outputs (False Article)": number of computed outputs for false article comparison. This should also sum up to all cases (in this case, 7).
"""
    question_list = generate_one_by_one_prompts(general_topics)
    data = []
    for i, list_of_data in enumerate(list_of_changed_articles):
        num_of_false = 0
        num_of_true = 0
        information = []

        if print_comments:
            print('True article:')
            print("Article: ", list_of_data[3])
            print("Facts: \n", list_of_data[4])

        for  fact in question_list:
            prompt_variables = {
                'article': list_of_data[3],
                'meaning_of_topic': fact["Description of fact"],
                'question': fact["Question for fact"].format(find_json(list_of_data[4])[0][fact["Name of fact"]]),
                'topic': fact["Description of fact"]
            }
            if testing:
                generated = testing_dictionary['labeling_true']
            else:
                generated = model_response(model_name, GENERAL_PROMPT_ONE_BY_ONE , prompt_variables)

            if print_comments == True:
                print("This was generated when comparing topic: ", fact["Description of fact"], " for the true article.")
                print(generated)

            opinion = find_first_true_or_false(generated)
            if opinion == "the answer is true":
                num_of_true += 1
            elif opinion == "the answer is false":
                num_of_false += 1
            else:
                pass
        print('True article:')
        if num_of_false + num_of_true == len(question_list):
            print(num_of_false)
            print(data)
            first_information = [num_of_false, num_of_false + num_of_true]
        else:
            print(num_of_false, "Missing!!!")
            first_information = [num_of_false, num_of_false + num_of_true]

        num_of_false = 0
        num_of_true = 0

        if print_comments:
            print("False article")
            print("Article: ", list_of_data[1])
            print("We changed", list_of_data[5], " and ", list_of_data[6])
            print("Fasts: \n", list_of_data[2])

        for  fact in question_list:
            prompt_variables = {
                'article': list_of_data[3],
                'meaning_of_topic': fact["Description of fact"],
                'question': fact["Question for fact"].format(find_json(list_of_data[4])[0][fact["Name of fact"]]),
                'topic': fact["Description of fact"]
                }
            
            if testing:
                generated = testing_dictionary['labeling_true']
            else:
                generated = model_response(model_name, GENERAL_PROMPT_ONE_BY_ONE , prompt_variables)

            if print_comments == True:
                print("This was generated when comparing topic: ", fact["Description of fact"], ", with the false article.")
                print(generated)
    
            opinion = find_first_true_or_false(generated)
            if opinion == "the answer is true":
                num_of_true += 1
            elif opinion == "the answer is false":
                num_of_false += 1
            else:
                pass
        print('False article:')
        if num_of_false + num_of_true == len(question_list):
            print(num_of_false)
            data.append([first_information[0], first_information[1], num_of_false, num_of_false + num_of_true])
        else:
            print(num_of_false, "Missing!!!")
            data.append([first_information[0], first_information[1], num_of_false, num_of_false + num_of_true])
    
    columns = ["False Negatives" ,  "Computed Outputs (True Article)",   "True Negatives", "Computed Outputs (False Article)"] 
    return  pd.DataFrame(data, columns=columns)

def fake_detect_comparison_true_to_true_and_false_to_true_changed(dict_of_changed_articles, model_name="llama3.1:8b", print_comments=False, testing=False): 
    """
    Inputs
    dict_of_changed_articles: A dictionary containing metadata of changed articles.
    Each entry has the structure:
    {
    'Index': index_value,
    'Original_article': "True Article Content",
    'Original_json': "Facts table",
    'Changed_article': "Fake article content",
    'Changed_json': "Changed facts",
    'Changed_topics': [<Name of fact>, <Name of fact>, ...]  # A list of topics that were changed,
    'topics': [
      {
        "Name of fact": "Turnout",
        "Description of fact": "The number of people who take part in election.",
        "Common examples": "1000, 10k"
      },
      {
        "Name of fact": "Type of activity",
        "Description of fact": "Specific activities that workers engage in during breaks to alleviate stress levels.",
        "Common examples": "Playing video games, Guided relaxation session, Staying silent"
      }
    ]
    model_name: The name of the model to be used for comparison (default is "llama3.1:8b").
    print_comments: A boolean flag to print intermediate comments (default is False).
    
    Outputs:
    Returns a pandas DataFrame containing the counts of true and false responses for each article pair. The DataFrame will look like this:
    
    Example:
    |   False Negatives |   Computed Outputs (True Article) |   True Negatives |   Computed Outputs (False Article) |
    |-------------------|-----------------------------------|------------------|------------------------------------|
    |                 0 |                                 7 |                2 |                                  7 |
    |                 1 |                                 7 |                3 |                                  7 |
    
    - "False Negatives": number of false negatives (true article is compared to true facts, so ideally this should sum to 0).
    - "Computed Outputs (True Article)": number of computed outputs for true article comparison. This should sum up to all cases (in this case, 7).
    - "True Negatives": number of true negatives (false article is compared to true facts, so ideally this should sum to 2).
    - "Computed Outputs (False Article)": number of computed outputs for false article comparison. This should also sum up to all cases (in this case, 7).
    """

    data = []

    for index, article_data in dict_of_changed_articles.iterrows():
        general_topics = article_data['topics']
        question_list = generate_one_by_one_prompts(general_topics)

        num_of_false = 0
        num_of_true = 0
        information = []

        if print_comments:
            print(f"Processing article at index {index}")
            print("True article:")
            print("Article: ", article_data['Original_article'])
            print("Facts: \n", article_data['Original_json'])

        # Process true article
        for fact in question_list:
            prompt_variables = {
                'article': article_data['Original_article'],
                'meaning_of_topic': fact["Description of fact"],
                'question': fact["Question for fact"].format(json.loads(article_data['Original_json'])[fact["Name of fact"]]),
                'topic': fact["Description of fact"]
            }
            if testing:
                generated = testing_dictionary['labeling_true']
            else:
                generated = model_response(model_name, GENERAL_PROMPT_ONE_BY_ONE, prompt_variables)

            if print_comments:
                print(f"This was generated when comparing topic: {fact['Description of fact']} for the true article:")
                print(generated)

            opinion = find_first_true_or_false(generated)
            if opinion == "the answer is true":
                num_of_true += 1
            elif opinion == "the answer is false":
                num_of_false += 1

        print('True article results:')
        if num_of_false + num_of_true == len(question_list):
            print(f"False answers: {num_of_false}")
            first_information = [num_of_false, num_of_false + num_of_true]
        else:
            print(f"{num_of_false} answers are missing!")
            first_information = [num_of_false, num_of_false + num_of_true]

        # Process false article
        if print_comments:
            print("False article:")
            print("Article: ", article_data['Changed_article'])
            print("Changed topics: ", article_data['topics_changed'])
            print("Facts: \n", article_data['Changed_json'])

        num_of_true = 0
        num_of_false = 0
        for fact in question_list:
            prompt_variables = {
                'article': article_data['Changed_article'],
                'meaning_of_topic': fact["Description of fact"],
                'question': fact["Question for fact"].format(json.load(article_data['Original_json'])[fact["Name of fact"]]),
                'topic': fact["Description of fact"]
            }
            if testing:
                generated = testing_dictionary['labeling_true']
            else:
                generated = model_response(model_name, GENERAL_PROMPT_ONE_BY_ONE, prompt_variables)

            if print_comments:
                print(f"This was generated when comparing topic: {fact['Description of fact']} for the false article:")
                print(generated)

            opinion = find_first_true_or_false(generated)
            changed_topics = article_data['topics_changed']  # Access the list of changed topics

            # Check if the current fact is in the changed topics list
            if (opinion == "the answer is true" and fact["Description of fact"] not in changed_topics) or (opinion == "the answer is false" and fact["Description of fact"] in changed_topics):
                num_of_true += 1
            elif (opinion == "the answer is true" and fact["Description of fact"] in changed_topics) or (opinion == "the answer is false" and fact["Description of fact"] not in changed_topics):
                num_of_false += 1

        print('False article results:')
        if num_of_false + num_of_true == len(question_list):
            print(f"False answers: {num_of_false}")
            data.append([first_information[0], first_information[1], num_of_false, num_of_false + num_of_true])
        else:
            print(f"{num_of_false} answers are missing!")
            data.append([first_information[0], first_information[1], num_of_false, num_of_false + num_of_true])

    columns = ["False Negatives" ,  "Computed Outputs (True Article)",   "True Negatives", "Computed Outputs (False Article)"] 
    return  pd.DataFrame(data, columns=columns)

def computation_of_similarity_and_correctness(dict_of_changed_articles, model_name="llama3.1:8b", print_comments=False, testing=False): 
    """
    Inputs
    dict_of_changed_articles: A dictionary containing metadata of changed articles.
    Each entry has the structure:
    {
    'Index': index_value,
    'Original_article': "True Article Content",
    'Original_json': "Facts table",
    'Changed_article': "Fake article content",
    'Changed_json': "Changed facts",
    'Changed_topics': [<Name of fact>, <Name of fact>, ...],  # A list of topics that were changed,
    'topics': [
      {
        "Name of fact": "Turnout",
        "Description of fact": "The number of people who take part in election.",
        "Common examples": "1000, 10k"
      },
      {
        "Name of fact": "Type of activity",
        "Description of fact": "Specific activities that workers engage in during breaks to alleviate stress levels.",
        "Common examples": "Playing video games, Guided relaxation session, Staying silent"
      }
    ]
    }
    model_name: The name of the model to be used for comparison (default is "llama3.1:8b").
    print_comments: A boolean flag to print intermediate comments (default is False).
    
    Outputs:
    Returns a pandas DataFrame containing the counts of true and false responses for each article pair. The DataFrame will look like this:
    
    Example:
    |   True Negatives |   Computed Outputs 1|   True Positives |   Computed Outputs 2|
    |------------------|---------------------|-------------------|---------------------|
    |                0 |                   2 |                 2 |                   2 |
    |                1 |                   2 |                 1 |                   2 |
    
    - "True Negatives": Number of correct identifications where the model correctly identifies false facts in the changed article.
    - "Computed Outputs": Total number of outputs computed for each article comparison (should be equal to number of changed facts). 
    - "True Positives": Number of examples, the model correctly Identifies the informations as overlapping.
    """
    data = []

    for index, article_data in dict_of_changed_articles.iterrows():
        general_topics = article_data['topics']
        id = article_data['id']
        question_list = generate_one_by_one_prompts(general_topics)
        question_list_2 = generate_one_by_one_prompts_2(general_topics)
        changed_topics = article_data['topics_changed']  # Access the list of changed topics

        if print_comments:
            print(f"\n\n\n ______________________________________________________________________________\nProcessing article with id {id}")
            
            print("True facts: \n", article_data['Original_json'])
            print("False facts: \n", article_data['Changed_json'])

        # Process true article
        num_of_false_negatives = 0
        num_of_true_negative = 0

        for i, fact in enumerate(question_list_2):
            if fact['Name of fact'] in changed_topics:
                prompt_variables = None
                first_fact = json.loads(article_data['Changed_json'])[fact["Name of fact"]]
                second_fact = json.loads(article_data['Original_json'])[fact["Name of fact"]]
                question = fact["Question for fact"].format(first_fact, second_fact)
                if testing:
                    generated = testing_dictionary['labeling_true']
                else:
                    generated = model_response(model_name, question, prompt_variables)

                if print_comments:
                    print(f'\n\n Fact {i}')
                    print(f"This was generated when comparing topic: {fact['Name of fact']} (we are comparing true and fake messages):")
                    print(generated)

                opinion = find_first_true_or_false(generated)

                if opinion == "the answer is true": #Facts have the same meaning
                    num_of_false_negatives += 1
                elif opinion == "the answer is false":
                    num_of_true_negative += 1

        total_outputs_facts = num_of_false_negatives + num_of_true_negative

        if print_comments:
            print('True article results:')
            print(f"False Negatives: {num_of_true_negative}")

        # Process false article
        if print_comments:
            print("False article: \n", article_data['Changed_article'])
            print("Changed topics: \n", article_data['topics_changed'])
            print("Changed facts: \n", article_data['Changed_json'])

        num_of_false_negative_2 = 0
        num_of_true_positive_2 = 0

        for i, fact in enumerate(question_list):
            if fact["Name of fact"] in changed_topics:
                prompt_variables = {
                    'article': article_data['Changed_article'],
                    'meaning_of_topic': fact["Description of fact"],
                    'question': fact["Question for fact"].format(json.loads(article_data['Changed_json'])[fact["Name of fact"]]),
                    'topic': fact["Description of fact"]
                }
                if testing:
                    generated = testing_dictionary['labeling_true']
                else:
                    formatted_content = GENERAL_PROMPT_ONE_BY_ONE.format(**prompt_variables)
                    print('\n\n\n\n this is the question: \n ', formatted_content)
                    generated = model_response(model_name, GENERAL_PROMPT_ONE_BY_ONE, prompt_variables)

                if print_comments:
                    print(f'\n\n Fact {i}')
                    print(f"This was generated when comparing topic: {fact['Description of fact']} in changed facts for the false article:")
                    print(generated)

                opinion = find_first_true_or_false(generated)

                if fact["Name of fact"] in changed_topics:
                    # The fact has been changed; it's false in the false article
                    if opinion == "the answer is false":
                        num_of_false_negative_2 += 1
                    elif opinion == "the answer is true":
                        num_of_true_positive_2 += 1

        total_outputs_false_article = num_of_false_negative_2 + num_of_true_positive_2

        if print_comments:
            print('False article results:')
            print(f"True Negatives: {num_of_true_positive_2}")

        data.append([num_of_true_negative, total_outputs_facts, num_of_true_positive_2, total_outputs_false_article])

    columns = ["True Negatives", "Computed Outputs 1", "False Negatives", "Computed Outputs 2"]
    return pd.DataFrame(data, columns=columns)


def fake_detect_only_for_one_example(data, general_topics, model_name = "llama3.1:8b", print_comments = False, testing = False):
    """Summary
    This function, fake_detect_only_for_one_example_strict, evaluates a given fake article by comparing it to predefined topics and their descriptions. The goal is to determine if the content of the fake article aligns with the expected descriptions. The function uses a language model to generate responses and evaluates whether the generated content matches "true" or "false" statements. Ideally, we want the response to be labeled as "true", indicating that the new article agrees with new facts.

    Example Usage
    data = {
        "topic": "Name of casualty or group",
        "fake_article": "The article content here...",
        "previous_json_file": {
            "Name of casualty or group": "Expected description here..."
        }
    }
    result = fake_detect_only_for_one_example_strict(data, general_topics, model_name="llama3.1:8b", print_comments=True)
    print(result)  # Expected output: {'Labelled': True} or {'Labelled': False} or {'Labelled': None}

    Code Analysis
    Inputs
    data: A dictionary containing the topic, fake article content, and previous JSON file with expected descriptions.
    general_topics: A list of topics containing the name, description, and question for each fact to compare.
    model_name: The name of the language model to use (default is "llama3.1:8b").
    print_comments: A boolean flag to print generated comments (default is False).
    testing: A flag to indicate whether to use pre-defined test data for evaluation (default is False).

    Flow
    1. Generate prompts based on the provided general topics.
    2. Iterate through the topics to compare the relevant one with the fake article.
    3. Generate a response using the language model for the current topic.
    4. If print_comments is enabled, print the generated response and the topic being compared.
    5. Evaluate the response for "true" or "false" and return the appropriate label:
       - {'Labelled': True} if the article matches the expected content.
       - {'Labelled': False} if the article contradicts the expected content.
       - {'Labelled': None} if no clear determination can be made.
       
    Outputs
    A dictionary containing the detection result: 
    - {'Labelled': True} if the article is consistent with the expected descriptions.
    - {'Labelled': False} if the article is inconsistent.
    - {'Labelled': None} if no decision could be made.
    """
    question_list = generate_one_by_one_prompts(general_topics)

    num_of_true = 0
    num_of_false = 0
    changed_topics = [data['topic']]
    for fact in question_list:
        if fact["Name of fact"] in changed_topics:
            prompt_variables = {
                'article': data["fake_article"],
                'meaning_of_topic': fact['Description of fact'],
                'question': fact['Question for fact'].format(data["changed_json_file"][fact["Name of fact"]]),
                'topic': fact["Name of fact"]
            }
            if testing:
                generated = testing_dictionary['labeling_true']
            else:
                generated = model_response(model_name, GENERAL_PROMPT_ONE_BY_ONE, prompt_variables)

#        for  fact in question_list:
#            prompt_variables = {
#                'article': data["fake_article"],
#                'meaning_of_topic': fact['Description of fact'],
#                'question': fact['Question for fact'].format(data["true_json_file"][fact["Name of fact"]]),
#                'topic': fact["Name of fact"]
#            }
#            if testing:
#                generated = testing_dictionary['labeling_true']
#            else:
#                generated = model_response(model_name, GENERAL_PROMPT_ONE_BY_ONE, prompt_variables)


            if print_comments == True:
                print("This was generated when comparing topic: ", fact["Name of fact"], ", with the false article.")
                print(generated)

            opinion = find_first_true_or_false(generated)
            print("False article:\n")
            changed_topics = [data['topic']]
            if opinion == "the answer is true":
                
                print("It is labeled true!")
                return {'Labelled': True }
            elif opinion == "the answer is false":
                print("It is labeled false!")
                return {'Labelled': False }
            else:
                print( "Missing!!!")
                return {'Labelled': None }
    return {'Labelled': None }

def fake_detect_only_for_one_example_strict(data, general_topics, model_name = "llama3.1:8b", print_comments = False, testing = False):
    """Summary
    This function, fake_detect_only_for_one_example_strict, evaluates a given fake article by comparing it to predefined topics and their descriptions. The goal is to determine if the content of the fake article aligns with the expected descriptions. The function uses a language model to generate responses and evaluates whether the generated content matches "true" or "false" statements. Ideally, we want the response to be labeled as "false," indicating that the fake article differs from the expected content.

    Example Usage
    data = {
        "topic": "Name of casualty or group",
        "fake_article": "The article content here...",
        "previous_json_file": {
            "Name of casualty or group": "Expected description here..."
        }
    }
    result = fake_detect_only_for_one_example_strict(data, general_topics, model_name="llama3.1:8b", print_comments=True)
    print(result)  # Expected output: {'Labelled': True} or {'Labelled': False} or {'Labelled': None}

    Code Analysis
    Inputs
    data: A dictionary containing the topic, fake article content, and previous JSON file with expected descriptions.
    general_topics: A list of topics containing the name, description, and question for each fact to compare.
    model_name: The name of the language model to use (default is "llama3.1:8b").
    print_comments: A boolean flag to print generated comments (default is False).
    testing: A flag to indicate whether to use pre-defined test data for evaluation (default is False).

    Flow
    1. Generate prompts based on the provided general topics.
    2. Iterate through the topics to compare the relevant one with the fake article.
    3. Generate a response using the language model for the current topic.
    4. If print_comments is enabled, print the generated response and the topic being compared.
    5. Evaluate the response for "true" or "false" and return the appropriate label:
       - {'Labelled': True} if the article matches the expected content.
       - {'Labelled': False} if the article contradicts the expected content.
       - {'Labelled': None} if no clear determination can be made.
       
    Outputs
    A dictionary containing the detection result: 
    - {'Labelled': True} if the article is consistent with the expected descriptions.
    - {'Labelled': False} if the article is inconsistent.
    - {'Labelled': None} if no decision could be made.
    """
    question_list = generate_one_by_one_prompts(general_topics)

    num_of_true = 0
    num_of_false = 0
    changed_topics = [data['topic']]
    for fact in question_list:
        if fact["Name of fact"] in changed_topics:
            prompt_variables = {
                'article': data["fake_article"],
                'meaning_of_topic': fact['Description of fact'],
                'question': fact['Question for fact'].format(data["previous_json_file"][fact["Name of fact"]]),
                'topic': fact["Name of fact"]
            }
            if testing:
                generated = testing_dictionary['labeling_false']
            else:
                generated = model_response(model_name, GENERAL_PROMPT_ONE_BY_ONE, prompt_variables)



            if print_comments == True:
                print("This was generated when comparing topic: ", fact["Name of fact"], ", with the false article.")
                print(generated)

            opinion = find_first_true_or_false(generated)
            print("False article:\n")
            changed_topics = [data['topic']]
            if opinion == "the answer is true":
                
                print("It is labeled true!")
                return {'Labelled': True }
            elif opinion == "the answer is false":
                print("It is labeled false!")
                return {'Labelled': False }
            else:
                print( "Missing!!!")
                return {'Labelled': None }
    return {'Labelled': None }
