import sys
import os
import random
import ollama
import json
import re
import pandas as pd
import General_news_labeling

current_path = os.getcwd()  # Get the current working directory
parent_directory = os.path.dirname(current_path)
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


GENERAL_CHANGES = """Dascription of \"{name_of_fact}\": {description_of_fact} You need to change some phrases in facts about \"{name_of_fact}\". 
            Changed values must bear different meaning but must not change the context.
            EXAMPLE OF CHANGED PHRASES:
                BEGINNING OF FACTS
                The research involved 66 workers who were divided into three groups for the experiment.
                END OF FACTS 
            CHANGED FACT
                A total of 200 employees from a local factory participated in the study, split into three distinct groups to analyze various types of activities.
             
            Common examples are: {common_examples}, but try to formulate your own example!"""

GENERAL_CHANGES_CHANGE_MEANING = """Dascription of \"{name_of_fact}\": {description_of_fact} You need to change some phrases or words in facts about \"{name_of_fact}\". 
            Changed phrases or words must heve different or opposite meaning! Maybe choose some other word to change the message of phrase.

            Common examples are: {common_examples}, but try to formulate your own example that bear different interpretation!"""

GENERAL_CHANGES_FUNNY = """Dascription of \"{name_of_fact}\": {description_of_fact} You need to change some phrases in facts about \"{name_of_fact}\". 
            Changed values must bear totally different meaning. 
            EXAMPLE OF CHANGED PHRASES:
                BEGINNING OF FACTS
                - Kafr Nabudah (northern countryside of Hama)\n- Idlib Province\n- Masqan village (northern countryside of Aleppo)
                END OF FACTS 
            CHANGED FACT
                - Ehras village (northern countryside of Aleppo)\n- Idlib Province\n- Masqan village (northern countryside of Aleppo)
             
            Common examples are: {common_examples}, but try to formulate your own example!"""


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

GENERAL_INFORMATION = "FAK-ES is a set of articles from Syrian war."

PROMPT_FOR_CORRECTING_MISTAKES_WHEN_CHANGING_FACTS = """You are trying to change the fact relating to {name_of_fact} to have the following information: {new_fact}. Read the following text:

BEGINNING_OF_THE_TEXT
{article_in_process_of_being_changed}
END_OF_THE_TEXT

Your task is to change the fact relating to {name_of_fact}. The original fact was {original_fact}. This original fact was supposed to be changed to {new_fact}. New text should begin with the phrase: BEGINNING_OF_THE_TEXT and end with the phrase END_OF_THE_TEXT."""

GENERAL_PROMPT_FACT_EXTRACTION = """You are a journalist tasked with analyzing an article that reports on casualties related to the war in Syria. Your goal is to extract specific information regarding casualties mentioned in the article.

    Please extract the following details of casualties in the news in JSON format.
    {{
        "Name of casualty or group": The individual's names or the names of the groups associated with the casualties.,
        "Gender or age group": Indicates if the persons is male or female, or specify their age groups (e.g., child, adult, senior).,
        "Cause of death": (e.g., shooting, shelling, chemical weapons, etc.),
        "Type": Classify the casualties as a civilian or non-civilian (e.g., military personnel are civilian).,
        "Actor": Identify the actors involved in the incidents, such as rebel groups, Russian forces, ISIS, the Syrian army, U.S. military, etc.,
        "Place of death": Specify the locations of the attack (e.g., Aleppo, Damascus, Homs, Idlib, Raqqa, Daraa, Deir ez-Zor, Qamishli, Palmyra, etc.).,
        "Date of death": Provide the dates when the attacks occurred.
    }}

    BEGINNING OF THE ARTICLE
    {article}
    END OF THE ARTICLE

    Ensure that the extracted information is as accurate and detailed as possible. Take context into account, and if certain data points are not available or mentioned in the article, output "Not available". Try to incorporate all casualties in one file.
    """

GENERAL_PROMPT_FOR_ONE_BY_ONE_FACT_EXTRACTION_CONCISE_VERSION = """You are an journalist with news articles. {general_information} Here is an article:
    BEGINNING OF THE ARTICLE
    {article}
    END OF THE ARTICLE

    Please extract the facts about the {topic} from the article. {topic_content}. Be as detailed as possible and include all information related to {topic}. Include only concise facts and do not output information that does not relate to {topic}, and if information is not present in the article, output "No information". Output should not exceed 25 words and should be written in the following format:
    BEGINNING OF FACTS
    <Display the extracted facts>
    END OF FACTS
    """

GENERAL_CHANGE_ONE_INFORMATION = {
    "paraphrase": """You have an article. {general_information}:

    BEGINNING OF THE ARTICLE
    {article}
    END OF THE ARTICLE

    Here is the related data extracted from the article in JSON format:

    {facts}

    Please, follow the instructions:
    Point 1: {changing_orders}
    Point 2: Create a new JSON file, which is the same as the old one, with the exception of {Name_of_fact}, which is given new information given in Point 1. JSON file should be displayed in standard notation, with use of double and not single quotes, and should contain all the key values as the original one.
    Point 3: Paraphrase a new article in which you will change the information for {Name_of_fact} according to decision made in Point 1. Ensure that all occurrences of {Name_of_fact} are changed. You must preserve all other information from article. This article should begin with the phrase "BEGINNING OF THE ARTICLE" and end with "END OF THE ARTICLE". Make sure you include those phrases.""",

    "paraphrase_aggressive": """To test your capabilities, I will give you a texts. {general_information}

    BEGINNING OF THE TEXT
    {article}
    END OF THE TEXT

    Here is the related data extracted from the text in JSON format:

    {facts}

    Please, follow the instructions:
    Point 1: {changing_orders}
    Point 2: Create a new JSON file, which is the same as the old one, with the exception of {Name_of_fact}, which is given new information given in Point 1. JSON file should be displayed in standard notation, with use of double and not single quotes, and should contain all the key values as the original one.
    Point 3: Paraphrase a new text in which you will change the information for {Name_of_fact} according to decision made in Point 1. Ensure that all occurrences of {Name_of_fact} are changed and included in new text. You must preserve all other information from text. This text should begin with the phrase "BEGINNING OF THE TEXT" and end with "END OF THE TEXT". Make sure you include those phrases.""",

    "summarize_aggressive": """To test your capabilities, I will give you a texts. {general_information} Here is an text:

    BEGINNING OF THE TEXT
    {article}
    END OF THE TEXT

    Here is the related facts extracted from the text in JSON format:

    {facts}

    Please, follow the instructions:
    Point 1: {changing_orders}
    Point 2: Create a new JSON file, which is the same as the old one, with the exception of {Name_of_fact}, which is given new information given in Point 1. JSON file should be displayed in standard notation, with use of double and not single quotes, and should contain all the key values as the original one.
    Point 3: Summarize the new text in which you will change the information for {Name_of_fact} according to decision made in Point 1. Ensure that all occurrences of {Name_of_fact} are changed and included in new summarization. You must preserve all other facts mentioned in the list of facts. This summarization should begin with the phrase "BEGINNING OF THE TEXT" and end with "END OF THE TEXT". Make sure you include those phrases."""
    ,
    "paraphrase_change_only_part_of_fact":"""You have an article, here is some general information about it : {general_information}:

    BEGINNING OF THE ARTICLE
    {article}
    END OF THE ARTICLE

    Here is the related data extracted from the article in JSON format:

    {facts}

    Please, follow the instructions:
    Point 1: {changing_orders}
    Point 2: Create a new JSON file, which is the same as the old one, with the exception of {Name_of_fact}, which is given new information given in Point 1. JSON file should be displayed in standard notation, with use of double and not single quotes, and should contain all the key values as the original one.
    Point 3: Paraphrase a new article in which you will change the information for {Name_of_fact} according to decision made in Point 1. Ensure that all occurrences of {Name_of_fact} are changed and included in new article. You must preserve all other information from article. This article should begin with the phrase "BEGINNING OF THE ARTICLE" and end with "END OF THE ARTICLE". Make sure you include those phrases.""",

}























#___________________________________________________________________________________________________________________________________________
# Function used for article generation


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

    # List of patterns to apply
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


def model_response(model_name, prompt, prompt_variables):
    formatted_content = prompt.format(**prompt_variables)
    response = ollama.chat(model=model_name, messages=[
        {
            'role': 'user',
            'content': formatted_content,
            'temperature': 0.2,
        }
    ])
    return response['message']['content']

def transform_topics_to_fact_extraction_prompt(topics, general_information, article):
    # Create a dictionary to store the transformed data
    transformed_dict = {}
    
    # Iterate over the topics and fill the transformed dictionary
    for topic in topics:
        ## Extract the name and description
        name_of_fact = topic.get('Name of fact', 'Unknown Fact')
        description = topic.get('Description of fact', 'Description not available.')
        common_examples = topic.get('Common examples', 'No common examples provided.')
        
        # Combine description with common examples
        transformed_description = f"{description.strip()} Common examples are: {common_examples.strip()}."
        
        # Assign the transformed description to the corresponding name
        transformed_dict[name_of_fact] = transformed_description
    
    # Create the formatted string representation
    formatted_string = f"""You are a journalist tasked with analyzing some articles. {general_information} Your goal is to extract specific information regarding casualties mentioned in the article.

    Please extract the following details of casualties in the news in JSON format.
    {{{{\n"""
    for key, value in transformed_dict.items():
        formatted_string += f'    "{key}": {value},\n'
    formatted_string = formatted_string.rstrip(",\n")  # Remove the trailing comma and newline
    formatted_string += f"""\n}}}}

    BEGINNING OF THE ARTICLE
    {article}
    END OF THE ARTICLE

    Ensure that the extracted information is as accurate and detailed as possible. Take context into account, and if certain data points are not available or mentioned in the article, output "Not available". Try to incorporate all casualties in one file.
    """

    return formatted_string

#Generation of text
def generate_facts_normal(article_content, name_of_the_model, print_generated_text = False):
    """
    Summary
    This function generates facts about casualties from a given article using a specified model. It formats the article content into a prompt, sends it to the model, and extracts the relevant facts in JSON format.

    Example Usage
    article_content = "An article about a recent conflict..."
    model_name = "fact_extraction_model"
    facts = generate_facts(article_content, model_name, print_generated_text=True)
    print(facts)
    Copy
    Insert

    Code Analysis
    Inputs
    article_content: The content of the article to analyze.
    name_of_the_model: The name of the model used for fact extraction.
    print_generated_text: A boolean flag to print the generated text (default is False).
    Flow
    The function formats the article content into a prompt using prompt_fact_extraction_abdul.
    It sends the prompt to the specified model using ollama.chat.
    The response from the model is processed to extract JSON-like content using find_json.
    If print_generated_text is True, it prints the generated text.
    The function returns the extracted JSON list or None if no facts are found.
    Outputs
    A list of dictionaries containing the extracted facts in JSON format, or None if no facts are found.
    """    
    prompt = transform_topics_to_fact_extraction_prompt(GENERAL_TOPICS, GENERAL_INFORMATION, article_content)
    print(prompt)
    prompt_variables = {}
    generated = model_response(name_of_the_model, prompt, prompt_variables)

    if print_generated_text == True:
        print("Facts generated in normal fact_generator:\n ", generated)

    json_list = find_json(generated)
    if json_list == []:
        json_list  = None
  
    return json_list

def generate_facts_one_by_one(article_content, general_topics, name_of_the_model, print_generated_text = False):
    """
    Summary
    This function generate_facts_one_by_one extracts specific facts from an article using a predefined list of topics. It uses a model to generate responses for each topic and compiles the extracted facts into a JSON list.

    Example Usage
    article_content = "Sample article content about the Syrian war."
    name_of_the_model = "fact_extraction_model"
    facts = generate_facts_one_by_one(article_content, name_of_the_model, print_generated_text=True)
    print(facts)

    Code Analysis
    Inputs
    article_content: The content of the article from which facts are to be extracted.
    name_of_the_model: The name of the model used for generating responses.
    print_generated_text: A boolean flag to print the generated text for each topic.
    Flow
    Initialize an empty dictionary json_dict.
    Iterate over each topic in the TOPICS list.
    Generate a response using the ollama.chat function with the formatted prompt.
    Optionally print the generated text if print_generated_text is True.
    Extract the fact from the generated text and store it in json_dict.
    Check if any value in json_dict is None and set json_list accordingly.
    Return the json_list.
    Outputs
    json_list: A list containing a dictionary of extracted facts, or None if any fact extraction failed.
    """
    json_dict = {}
    
    for topic in general_topics:
        keys_of_topic = [key for key in topic.keys()]
        if "Name of fact" in keys_of_topic and "Description of fact" in keys_of_topic and "Common examples" in keys_of_topic: #If topic is of the right format
            
            prompt_variables = {
                'article': article_content,
                'topic': topic["Name of fact"],
                'topic_content': topic["Description of fact"],
                'general_information': GENERAL_INFORMATION,
            }
            generated = model_response(name_of_the_model, GENERAL_PROMPT_FOR_ONE_BY_ONE_FACT_EXTRACTION_CONCISE_VERSION, prompt_variables)
        else:
            return None

        if print_generated_text == True:
            print(f"Generated fact on topic {topic['Name of fact']}:\n", generated)
        json_dict[topic["Name of fact"]] = extract_fact_from_text(generated)
        if None in list(json_dict.values()):
            json_list = None
        else: json_list = [json_dict]

    return json_list

def generate_facts(articles, general_topics, name_of_the_model = "Llama3.1:8B", size_of_sample = 5, type_of_generation = "normal", print_comments=False):
    """
    Summary
    This function generates facts from a sample of articles using a specified model. It can generate facts either in a "normal" mode or "one_by_one" mode and returns the results in a DataFrame.

    Example Usage
    articles = pd.DataFrame({'article_content': ["Article 1 content", "Article 2 content"]})
    model_name = "fact_extraction_model"
    sample_size = 1
    results = generate(articles, model_name, sample_size, type_of_generation="normal", print_comments=True)
    print(results)

    Inputs
    articles: DataFrame containing articles.
    name_of_the_model: The name of the model used for fact extraction.
    size_of_sample: Number of articles to sample.
    type_of_generation: Mode of fact generation ("normal" or "one_by_one").
    print_comments: Boolean flag to print generated text.
    Flow
    Randomly sample indices from the articles.
    For each sampled article, generate facts using the specified mode.
    Append the results to a list.
    Convert the results list to a DataFrame and return it.
    Outputs
    A DataFrame containing the indices, article contents, and extracted facts in JSON format.
    """
    
    indices = random.sample(list(range(len(articles))), size_of_sample)
    results = []
    for index in indices:
        article_content = list(articles['article_content'])[index]

        if type_of_generation == "normal":
            json_list = generate_facts_normal(article_content, name_of_the_model, print_comments)
        elif type_of_generation == "one_by_one":
            json_list = generate_facts_one_by_one(article_content, general_topics,  name_of_the_model, print_comments)
        else:
            raise ValueError("argument 'type of generation' is not a valid string!")

        if json_dict == None:
            print("No JSON found in first step of extracting facts!")
        else:
            for json_dict in json_list:
                results.append(index, article_content, json_dict)
    
    column_names = ["index", "article_content", "json_dict"]
    results = pd.DataFrame(results, columns = column_names)
    return results


















#________________________________________________________________
#main functions

def extract_and_change_articles(
    articles,
    general_topics,
    name_of_the_model,
    size_of_sample=5,
    type_of_generation="normal",
    change_of_article="paraphrase_aggressive",
    print_comments=False,
    testing=False,
    number_of_facts_changed=2  # Added the new argument with default value 2
):
    """
    Summary
    This function extracts and modifies articles from a given dataset using a specified model.
    It samples a subset of articles, generates facts, and changes specific topics within the articles.

    Inputs
    - articles: List or DataFrame containing articles.
    - general_topics: List of general topics for fact extraction.
    - name_of_the_model: Name of the model used for fact extraction and modification.
    - size_of_sample: Number of articles to sample (default is 5).
    - type_of_generation: Type of fact generation method ("normal" or "one_by_one").
    - change_of_article: Type of article change method.
    - print_comments: Boolean flag to print intermediate steps (default is False).
    - testing: Boolean flag for testing mode.
    - number_of_facts_changed: Number of facts to change in each article (default is 2).

    Outputs
    DataFrame containing the original and modified articles, along with the topics changed.
    """
    indices = random.sample(list(range(len(articles))), size_of_sample)
    results = []

    for index in indices:
        article_content = articles[index]
        if print_comments:
            print("Original article: ", article_content)

        if testing:
            json_list = testing_dictionary['fact_extraction']
        elif type_of_generation == "normal":
            json_list = generate_facts_normal(article_content, name_of_the_model, print_comments)
        elif type_of_generation == "one_by_one":
            json_list = generate_facts_one_by_one(
                article_content, general_topics, name_of_the_model, print_comments
            )
        else:
            raise ValueError("argument 'type of generation' is not a valid string!")

        if json_list is None:
            print("No JSON found in first step of extracting facts!")
        else:
            for json_dict in json_list:
                print("THIS IS THE DICTIONARY THAT WAS EXTRACTED")
                print_readable_dict(json_dict)

                change_topics_formulated = generate_aggressive_prompts(general_topics)
                # Use the new argument here
                topic_to_change = random.sample(change_topics_formulated, number_of_facts_changed)
                list_of_topics_changed = [topic["Name of fact"] for topic in topic_to_change]
                print("We will change topics:", ", ".join(list_of_topics_changed))

                current_article = article_content
                current_json = json_dict
                success = True

                # Loop over the topics to change
                for i, topic in enumerate(topic_to_change):
                    prompt_variables = {
                        'article': current_article,
                        'facts': json.dumps(current_json, indent=4, ensure_ascii=False),
                        'Name_of_fact': topic["Name of fact"],
                        'changing_orders': topic["Changing orders"],
                        'general_information': GENERAL_INFORMATION,
                    }
                    if testing:
                        generated = testing_dictionary['article_generation']
                    else:
                        generated = model_response(
                            name_of_the_model,
                            GENERAL_CHANGE_ONE_INFORMATION[change_of_article],
                            prompt_variables
                        )

                    changed_json = find_json(generated, general_topics)
                    changed_json = changed_json[0] if changed_json else None
                    changed_article = extract_last_text(generated)

                    if not changed_article or not changed_json:
                        print(f"We did not get through round {i + 1}!")
                        if print_comments:
                            print(f"THIS IS THE GENERATED TEXT IN ROUND {i + 1} \n\n", generated)
                            print("EXTRACTED ARTICLE: ", changed_article)
                            print("EXTRACTED JSON: ", changed_json)
                        success = False
                        break  # Exit the loop if the generation fails
                    else:
                        current_article = changed_article
                        current_json = changed_json
                        if print_comments:
                            print(f"THIS IS THE GENERATED TEXT IN ROUND {i + 1} \n\n", generated)
                            print(f"EXTRACTED ARTICLE IN ROUND {i + 1}: ", changed_article)
                            print(f"EXTRACTED JSON IN ROUND {i + 1}: ", changed_json)

                if success:
                    results.append([
                        index,
                        current_article,
                        json.dumps(current_json, indent=4, ensure_ascii=False),
                        article_content,
                        json.dumps(json_dict, indent=4, ensure_ascii=False),
                        list_of_topics_changed  # Store all topics changed
                    ])
                else:
                    results.append([
                        index,
                        None,
                        None,
                        article_content,
                        json_dict,
                        list_of_topics_changed  # Store attempted topics even if failed
                    ])

    column_names = [
        "index",
        "Changed_article",
        "Changed_json",
        "Original_article",
        "Original_json",
        "topics_changed"  # Updated column name
    ]
    results = pd.DataFrame(results, columns=column_names)
    return results


def extract_and_change_articles_with_labeling_added(
    articles,
    general_topics,
    name_of_the_model,
    type_of_generation="one_by_one",
    change_of_article="paraphrase_aggressive",
    number_of_changed_facts=3,
    print_comments=False,
    testing=False
):
    """
    Summary:
    This function processes a sample of articles to generate and modify facts using a specified model.
    It iteratively changes specified topics within the articles and evaluates the quality of these changes using a fake detector, to improve performance.

    Example Usage:
    ```python
    articles = pd.DataFrame({'article_content': ["Article 1 content...", "Article 2 content..."]})
    model_name = "fact_extraction_model"
    results = extract_and_change_articles_with_labeling_added(
        articles=articles,
        general_topics=GENERAL_TOPICS,
        name_of_the_model=model_name,
        size_of_sample=2,
        type_of_generation="normal",
        change_of_article="paraphrase_abdul",
        number_of_changed_facts=3,
        print_comments=True
    )
    print(results)
    ```

    Inputs:
    - articles: List or DataFrame containing articles to process.
    - general_topics: List of dictionaries defining general topics.
    - name_of_the_model: Name of the model used for fact extraction and modification.
    - size_of_sample: Number of articles to sample from articles (default is 5).
    - type_of_generation: Method of fact generation, either "normal" or "one_by_one".
    - change_of_article: Type of article modification to apply.
    - number_of_changed_facts: Number of facts/topics to change in the article (default is 3).
    - print_comments: Boolean flag to print intermediate comments (default is False).
    - testing: Boolean flag to indicate if testing mode is active (default is False).

    Flow:
    - Randomly sample articles from the provided articles.
    - Generate facts from each sampled article using the specified model.
    - Modify the article content iteratively by changing specified topics.
    - Evaluate the quality of the modified articles using a fake detector.
    - Collect and return the results in a DataFrame.

    Outputs:
    - DataFrame containing the original and modified articles, JSON data, and topics changed.
    """

    # Randomly select indices for sampling articles
    results = []

    #Return an empty DataFrame if the number of changed facts exceeds the number of general topics
    if number_of_changed_facts > len(general_topics): 
        print("\nNumber of changed facts exceeds the number of general topics! I could not proceed with the work.\n")
        return pd.DataFrame({})
    
    # Iterate over each selected article index
    for index, row in articles.iterrows():
        article_content = row["article_content"]
        if print_comments:
            print("Original article:", article_content)

        # Generate facts from the article
        if testing:
            json_list = testing_dictionary['fact_extraction']
        elif type_of_generation == "normal":
            json_list = generate_facts_normal(article_content, name_of_the_model, print_comments)
        elif type_of_generation == "one_by_one":
            json_list = generate_facts_one_by_one(
                article_content, general_topics, name_of_the_model, print_comments
            )
        else:
            raise ValueError("Argument 'type_of_generation' is not a valid string!")

        if json_list is None:
            print("No JSON found in the first step of extracting facts!")
        else:
            if print_comments:
                print("Extracted dictionary:")
                print_readable_dict(json_list)

            for json_dict in json_list:
                quality_of_generation = False
                number_of_trials = 0

                # Randomly select topics to change
                topics_to_change = random.sample(
                    generate_aggressive_prompts(general_topics),
                    number_of_changed_facts
                )
                topic_names = [topic["Name of fact"] for topic in topics_to_change]
                if print_comments:
                    print("We will change topics:", ", ".join(topic_names))

                changed_article = article_content
                changed_json = json_dict

                for i in range(number_of_changed_facts):
                    number_of_trials = 0
                    quality_of_generation = False
                    # Loop until a good quality generation is achieved or max trials are reached
                    while not quality_of_generation and number_of_trials <= 3:

                        number_of_trials += 1
                        name_of_changed_fact = topics_to_change[i]["Name of fact"]

                        # Iterate over the number of facts to change
                        if number_of_trials == 1: #This is the first trial
                            prompt_variables = {
                                'article': changed_article,
                                'facts': json.dumps(changed_json, indent=4, ensure_ascii=False),
                                'Name_of_fact': name_of_changed_fact,
                                'changing_orders': topics_to_change[i]["Changing orders"],
                                'general_information': "",
                            }
                            if testing:
                                generated = testing_dictionary['article_generation']
                            else:
                                generated = model_response(
                                    name_of_the_model,
                                    GENERAL_CHANGE_ONE_INFORMATION[change_of_article],
                                    prompt_variables
                                )
                        else:
                            new_fact = json_in_process_of_being_changed[name_of_changed_fact]
                            if print_comments:
                                print(f"Program failed to change the {name_of_changed_fact}")
                                print(f"The original article was {changed_article} \n Its {name_of_changed_fact} was supposed to be changed to {new_fact}")
                                print(f"LLMs failed to generate such article. The answer was: \n {article_in_process_of_being_changed}")

                            prompt_variables = {
                                'article_in_process_of_being_changed': article_in_process_of_being_changed,
                                'name_of_fact': name_of_changed_fact,
                                'original_fact': changed_json[name_of_changed_fact],
                                'new_fact': new_fact,
                            }
                            if testing:
                                generated = testing_dictionary['article_generation']
                            else:
                                generated = model_response(
                                    name_of_the_model,
                                    PROMPT_FOR_CORRECTING_MISTAKES_WHEN_CHANGING_FACTS,
                                    prompt_variables
                                )
                        # Extract the changed article and JSON
                        changed_json_list = find_json(generated, general_topics)
                        json_in_process_of_being_changed = changed_json_list[0] if changed_json_list else False
                        article_in_process_of_being_changed = extract_last_text(generated)
                        if not article_in_process_of_being_changed or not json_in_process_of_being_changed:
                            if print_comments:
                                print("Failure in detection.")
                                print("This was the generated content: \n", generated)
                            break  # Exit the loop if extraction failed
                        else:
                            # Evaluate the quality of the generated article
                            extracted = {
                                "fake_article": article_in_process_of_being_changed,
                                "changed_json_file": json_in_process_of_being_changed,
                                "topic": topic_names[i]
                            }
                            labeling_result = General_news_labeling.fake_detect_only_for_one_example(
                                extracted, general_topics= general_topics, print_comments=print_comments, testing=True
                            )

                            if print_comments: 
                                print(f"Final generated text in step {i}:\n", generated)
                                print(f"Extracted article in step {i}:", changed_article)
                                print(f"Extracted JSON at step {i}:", changed_json)
                    # Append the results

                            if labeling_result['Labelled']:
                                quality_of_generation = True
                                if print_comments:
                                    print("Quality of generation was good.")
                            else:
                                quality_of_generation = False
                                if print_comments:
                                    print(f"Quality of generation is bad! Label is equal to {labeling_result['Labelled']}.")
                                    print("Generated text:\n", generated)
                    
                                break  # Exit the loop if labeling failed
                    
                        
                    
                    changed_article = article_in_process_of_being_changed
                    changed_json = json_in_process_of_being_changed

                if quality_of_generation:
                    if print_comments:
                        print("The changed article was approved.")
                    # Append the results
                    results.append([
                        index,
                        changed_article,
                        json.dumps(changed_json, indent=4, ensure_ascii=False),
                        article_content,
                        json.dumps(json_dict, indent=4, ensure_ascii=False),
                        topic_names
                    ])
                else:
                    if print_comments:
                        print("Failed to generate a good quality article after maximum trials.")

    # Define column names for the results DataFrame
    column_names = [
        "index",
        "Changed_article",
        "Changed_json",
        "Original_article",
        "Original_json",
        "topics_changed"
    ]
    results = pd.DataFrame(results, columns=column_names)

    return results


def extract_and_change_articles_with_labeling_added_strict(
    articles,
    general_topics,
    name_of_the_model,
    type_of_generation="one_by_one",
    change_of_article="paraphrase_aggressive",
    number_of_changed_facts=3,
    print_comments=False,
    testing=False
):
    """
    Summary:
    This function processes a sample of articles to generate and modify facts using a specified model.
    It iteratively changes specified topics within the articles and evaluates the quality of these changes using a fake detector, to improve detector. Fake detector is used in a strict way.

    Example Usage:
    ```python
    articles = pd.DataFrame({'article_content': ["Article 1 content...", "Article 2 content..."]})
    model_name = "fact_extraction_model"
    results = extract_and_change_articles_with_labeling_added(
        articles=articles,
        general_topics=GENERAL_TOPICS,
        name_of_the_model=model_name,
        size_of_sample=2,
        type_of_generation="normal",
        change_of_article="paraphrase_abdul",
        number_of_changed_facts=3,
        print_comments=True
    )
    print(results)
    ```

    Inputs:
    - articles: List or DataFrame containing articles to process.
    - general_topics: List of dictionaries defining general topics.
    - name_of_the_model: Name of the model used for fact extraction and modification.
    - size_of_sample: Number of articles to sample from articles (default is 5).
    - type_of_generation: Method of fact generation, either "normal" or "one_by_one".
    - change_of_article: Type of article modification to apply.
    - number_of_changed_facts: Number of facts/topics to change in the article (default is 3).
    - print_comments: Boolean flag to print intermediate comments (default is False).
    - testing: Boolean flag to indicate if testing mode is active (default is False).

    Flow:
    - Randomly sample articles from the provided articles.
    - Generate facts from each sampled article using the specified model.
    - Modify the article content iteratively by changing specified topics.
    - Evaluate the quality of the modified articles using a fake detector.
    - Collect and return the results in a DataFrame.

    Outputs:
    - DataFrame containing the original and modified articles, JSON data, and topics changed.
    """

    # Randomly select indices for sampling articles
    results = []

    #Return an empty DataFrame if the number of changed facts exceeds the number of general topics
    if number_of_changed_facts > len(general_topics): 
        print("\nNumber of changed facts exceeds the number of general topics! I could not proceed with the work.\n")
        return pd.DataFrame({})

    # Iterate over each selected article index
    for index, row in articles.iterrows():
        article_content = row["article_content"]
        if print_comments:
            print("Original article:", article_content)

        # Generate facts from the article
        if testing:
            json_list = testing_dictionary['fact_extraction']
        elif type_of_generation == "normal":
            json_list = generate_facts_normal(article_content, name_of_the_model, print_comments)
        elif type_of_generation == "one_by_one":
            json_list = generate_facts_one_by_one(
                article_content, general_topics, name_of_the_model, print_comments
            )
        else:
            raise ValueError("Argument 'type_of_generation' is not a valid string!")

        if json_list is None:
            print("No JSON found in the first step of extracting facts!")
        else:
            if print_comments:
                print("Extracted dictionary:")
                print_readable_dict(json_list)

            for json_dict in json_list:
                quality_of_generation = False
                number_of_trials = 0

                 # Randomly select topics to change
                topics_to_change = random.sample(
                    generate_aggressive_prompts(general_topics),
                    number_of_changed_facts
                )
                topic_names = [topic["Name of fact"] for topic in topics_to_change]
                if print_comments:
                    print("We will change topics:", ", ".join(topic_names))

                changed_article = article_content
                changed_json = json_dict

                for i in range(number_of_changed_facts):
                    number_of_trials = 0
                    quality_of_generation = False
                    # Loop until a good quality generation is achieved or max trials are reached
                    while not quality_of_generation and number_of_trials <= 3:

                        number_of_trials += 1
                        name_of_changed_fact = topics_to_change[i]["Name of fact"]

                        # Iterate over the number of facts to change
                        if number_of_trials == 1: #This is the first trial
                            prompt_variables = {
                                'article': changed_article,
                                'facts': json.dumps(changed_json, indent=4, ensure_ascii=False),
                                'Name_of_fact': name_of_changed_fact,
                                'changing_orders': topics_to_change[i]["Changing orders"],
                                'general_information': "",
                            }
                            if testing:
                                generated = testing_dictionary['article_generation']
                            else:
                                generated = model_response(
                                    name_of_the_model,
                                    GENERAL_CHANGE_ONE_INFORMATION[change_of_article],
                                    prompt_variables
                                )
                        else:
                            new_fact = json_in_process_of_being_changed[name_of_changed_fact]
                            if print_comments:
                                print(f"Program failed to change the {name_of_changed_fact}")
                                print(f"The original article was {changed_article} \n Its {name_of_changed_fact} was supposed to be changed to {new_fact}")
                                print(f"LLMs failed to generate such article. The answer was: \n {article_in_process_of_being_changed}")

                            prompt_variables = {
                                'article_in_process_of_being_changed': article_in_process_of_being_changed,
                                'name_of_fact': name_of_changed_fact,
                                'original_fact': changed_json[name_of_changed_fact],
                                'new_fact': new_fact,
                            }
                            if testing:
                                generated = testing_dictionary['article_generation']
                            else:
                                generated = model_response(
                                    name_of_the_model,
                                    PROMPT_FOR_CORRECTING_MISTAKES_WHEN_CHANGING_FACTS,
                                    prompt_variables
                                )
                        # Extract the changed article and JSON
                        changed_json_list = find_json(generated, general_topics)
                        json_in_process_of_being_changed = changed_json_list[0] if changed_json_list else False
                        article_in_process_of_being_changed = extract_last_text(generated)
                        if not article_in_process_of_being_changed or not json_in_process_of_being_changed:
                            if print_comments:
                                print("Failure in detection.")
                                print("This was the generated content: \n", generated)
                            break  # Exit the loop if extraction failed
                        else:
                            # Evaluate the quality of the generated article
                            extracted = {
                                "fake_article": article_in_process_of_being_changed,
                                "previous_json_file": changed_json,
                                "topic": topic_names[i]
                            }
                            labeling_result = General_news_labeling.fake_detect_only_for_one_example_strict(
                                extracted, general_topics= general_topics, print_comments=print_comments, testing=True
                            )

                            if print_comments: 
                                print(f"Final generated text in step {i}:\n", generated)
                                print(f"Extracted article in step {i}:", changed_article)
                                print(f"Extracted JSON at step {i}:", changed_json)
                    # Append the results

                            if labeling_result['Labelled'] == False:
                                quality_of_generation = True
                                if print_comments:
                                    print("Quality of generation was good.")
                            else:
                                quality_of_generation = False
                                if print_comments:
                                    print(f"Quality of generation is bad! Label is equal to {labeling_result['Labelled']}.")
                                    print("Generated text:\n", generated)
                    
                                break  # Exit the loop if labeling failed
                    
                        
                    
                    changed_article = article_in_process_of_being_changed
                    changed_json = json_in_process_of_being_changed

                if quality_of_generation:
                    if print_comments:
                        print("The changed article was approved.")
                    # Append the results
                    results.append([
                        index,
                        changed_article,
                        json.dumps(changed_json, indent=4, ensure_ascii=False),
                        article_content,
                        json.dumps(json_dict, indent=4, ensure_ascii=False),
                        topic_names
                    ])
                else:
                    if print_comments:
                        print("Failed to generate a good quality article after maximum trials.")

    # Define column names for the results DataFrame
    column_names = [
        "index",
        "Changed_article",
        "Changed_json",
        "Original_article",
        "Original_json",
        "topics_changed"
    ]
    results = pd.DataFrame(results, columns=column_names)

    return results


def extract_and_change_articles_with_labeling_added_repeat(
    articles,
    general_topics,
    name_of_the_model,
    type_of_generation="one_by_one",
    change_of_article="paraphrase_aggressive",
    number_of_changed_facts=3,
    print_comments=False,
    testing=False
):
    """
    Summary:
    This function processes a sample of articles to generate and modify facts using a specified model.
    It iteratively changes specified topics within the articles and evaluates the quality of these changes using a fake detector, to improve performance.

    Example Usage:
    ```python
    articles = pd.DataFrame({'article_content': ["Article 1 content...", "Article 2 content..."]})
    model_name = "fact_extraction_model"
    results = extract_and_change_articles_with_labeling_added(
        articles=articles,
        general_topics=GENERAL_TOPICS,
        name_of_the_model=model_name,
        size_of_sample=2,
        type_of_generation="normal",
        change_of_article="paraphrase_abdul",
        number_of_changed_facts=3,
        print_comments=True
    )
    print(results)
    ```

    Inputs:
    - articles: List or DataFrame containing articles to process.
    - general_topics: List of dictionaries defining general topics.
    - name_of_the_model: Name of the model used for fact extraction and modification.
    - size_of_sample: Number of articles to sample from articles (default is 5).
    - type_of_generation: Method of fact generation, either "normal" or "one_by_one".
    - change_of_article: Type of article modification to apply.
    - number_of_changed_facts: Number of facts/topics to change in the article (default is 3).
    - print_comments: Boolean flag to print intermediate comments (default is False).
    - testing: Boolean flag to indicate if testing mode is active (default is False).

    Flow:
    - Randomly sample articles from the provided articles.
    - Generate facts from each sampled article using the specified model.
    - Modify the article content iteratively by changing specified topics.
    - Evaluate the quality of the modified articles using a fake detector.
    - Collect and return the results in a DataFrame.

    Outputs:
    - DataFrame containing the original and modified articles, JSON data, and topics changed.
    """

    # Randomly select indices for sampling articles
    results = []

    #Return an empty DataFrame if the number of changed facts exceeds the number of general topics
    if number_of_changed_facts > len(general_topics): 
        print("\nNumber of changed facts exceeds the number of general topics! I could not proceed with the work.\n")
        return pd.DataFrame({})
    
    # Iterate over each selected article index
    for index, row in articles.iterrows():
        article_content = row["article_content"]
        if print_comments:
            print("Original article:", article_content)

        # Generate facts from the article
        if testing:
            json_list = testing_dictionary['fact_extraction']
        elif type_of_generation == "normal":
            json_list = generate_facts_normal(article_content, name_of_the_model, print_comments)
        elif type_of_generation == "one_by_one":
            json_list = generate_facts_one_by_one(
                article_content, general_topics, name_of_the_model, print_comments
            )
        else:
            raise ValueError("Argument 'type_of_generation' is not a valid string!")

        if json_list is None:
            print("No JSON found in the first step of extracting facts!")
        else:
            if print_comments:
                print("Extracted dictionary:")
                print_readable_dict(json_list)

            for json_dict in json_list:
                quality_of_generation = False
                number_of_trials = 0

                # Randomly select topics to change
                topics_to_change = random.sample(
                    generate_aggressive_prompts(general_topics),
                    number_of_changed_facts
                )
                topic_names = [topic["Name of fact"] for topic in topics_to_change]
                if print_comments:
                    print("We will change topics:", ", ".join(topic_names))

                changed_article = article_content
                changed_json = json_dict

                for i in range(number_of_changed_facts):
                    number_of_trials = 0
                    quality_of_generation = False
                    # Loop until a good quality generation is achieved or max trials are reached
                    while not quality_of_generation and number_of_trials <= 3:

                        number_of_trials += 1
                        name_of_changed_fact = topics_to_change[i]["Name of fact"]

                        # Iterate over the number of facts to change
                        if number_of_trials == 1: #This is the first trial
                            prompt_variables = {
                                'article': changed_article,
                                'facts': json.dumps(changed_json, indent=4, ensure_ascii=False),
                                'Name_of_fact': name_of_changed_fact,
                                'changing_orders': topics_to_change[i]["Changing orders"],
                                'general_information': "",
                            }
                            if testing:
                                generated = testing_dictionary['article_generation']
                            else:
                                generated = model_response(
                                    name_of_the_model,
                                    GENERAL_CHANGE_ONE_INFORMATION[change_of_article],
                                    prompt_variables
                                )
                        else:
                            new_fact = json_in_process_of_being_changed[name_of_changed_fact]
                            if print_comments:
                                print(f"Program failed to change the {name_of_changed_fact}")
                                print(f"The original article was {changed_article} \n Its {name_of_changed_fact} was supposed to be changed to {new_fact}")
                                print(f"LLMs failed to generate such article. The answer was: \n {article_in_process_of_being_changed}")

                            prompt_variables = {
                                'article_in_process_of_being_changed': article_in_process_of_being_changed,
                                'name_of_fact': name_of_changed_fact,
                                'original_fact': changed_json[name_of_changed_fact],
                                'new_fact': new_fact,
                            }
                            if testing:
                                generated = testing_dictionary['article_generation']
                            else:
                                generated = model_response(
                                    name_of_the_model,
                                    PROMPT_FOR_CORRECTING_MISTAKES_WHEN_CHANGING_FACTS,
                                    prompt_variables
                                )
                        # Extract the changed article and JSON
                        changed_json_list = find_json(generated, general_topics)
                        json_in_process_of_being_changed = changed_json_list[0] if changed_json_list else False
                        article_in_process_of_being_changed = extract_last_text(generated)
                        if not article_in_process_of_being_changed or not json_in_process_of_being_changed:
                            if print_comments:
                                print("Failure in detection.")
                                print("This was the generated content: \n", generated)
                            break  # Exit the loop if extraction failed
                        else:
                            # Evaluate the quality of the generated article
                            extracted = {
                                "fake_article": article_in_process_of_being_changed,
                                "changed_json_file": json_in_process_of_being_changed,
                                "topic": topic_names[i]
                            }
                            labeling_result = General_news_labeling.fake_detect_only_for_one_example(
                                extracted, general_topics= general_topics, print_comments=print_comments, testing=True
                            )

                            if print_comments: 
                                print(f"Final generated text in step {i}:\n", generated)
                                print(f"Extracted article in step {i}:", changed_article)
                                print(f"Extracted JSON at step {i}:", changed_json)
                    # Append the results

                            if labeling_result['Labelled']:
                                quality_of_generation = True
                                if print_comments:
                                    print("Quality of generation was good.")
                            else:
                                quality_of_generation = False
                                if print_comments:
                                    print(f"Quality of generation is bad! Label is equal to {labeling_result['Labelled']}.")
                                    print("Generated text:\n", generated)
                    
                                break  # Exit the loop if labeling failed
                    
                        
                    
                    changed_article = article_in_process_of_being_changed
                    changed_json = json_in_process_of_being_changed

                if quality_of_generation:
                    if print_comments:
                        print("The changed article was approved.")
                    # Append the results
                    results.append([
                        index,
                        changed_article,
                        json.dumps(changed_json, indent=4, ensure_ascii=False),
                        article_content,
                        json.dumps(json_dict, indent=4, ensure_ascii=False),
                        topic_names
                    ])
                else:
                    if print_comments:
                        print("Failed to generate a good quality article after maximum trials.")

    # Define column names for the results DataFrame
    column_names = [
        "index",
        "Changed_article",
        "Changed_json",
        "Original_article",
        "Original_json",
        "topics_changed"
    ]
    results = pd.DataFrame(results, columns=column_names)

    return results