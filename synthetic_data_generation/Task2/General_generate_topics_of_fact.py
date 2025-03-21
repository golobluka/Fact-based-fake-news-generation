import ollama
import os
import json
import sys
import re
import pandas as pd
import random

current_path = os.getcwd()  # Get the current working directory
parent_directory = os.path.dirname(current_path)
sys.path.append(parent_directory)

# Attempted Import
try:
    from constants import (
        TESTING_TOPICS
    )
    print("Imported Constants Successfully.")
except ImportError as e:
    print("ImportError:", e)
    # Print available attributes from the constants module
    import constants
    print("Available Constants:", dir(constants))
    raise  # Raise the error after printing




# Prompts

prompt = """
You are a journalist working on world news. Extract {number_of_facts} different topics of facts from the given article.

BEGINNING OF THE ARTICLE
{article}
END OF THE ARTICLE

Topics should be outputted in standard JSON form as follows:
[
  {{
    "Name of fact": "name of type of fact",
    "Description of fact": "What information does the fact contain.",
    "Common examples": "Some examples of the facts of prescribed type."
  }}
]

BEGINNING OF THE ARTICLE
{article}
END OF THE ARTICLE

OUTPUT EXAMPLES OF TOPICS
[
  {{
    "Name of fact": "Turnout",
    "Description of fact": "The number of people who take part in election.",
    "Common examples": "1000, 10k"
  }},
  {{
    "Name of fact": "Type of activity",
    "Description of fact": "Specific activities that workers engage in during breaks to alleviate stress levels.",
    "Common examples": "Playing video games, Guided relaxation session, Staying silent"
  }},
  {{
    "Name of fact": "Impact on stress levels",
    "Description of fact": "How different types of activities affect the stress levels of workers.",
    "Common examples": "Increased worry and stress, Less worried and stressed, Much better than before"
  }},
  {{
    "Name of fact": "Number of participants",
    "Description of fact": "The number of workers who took part in the experiment to test different types of activities.",
    "Common examples": "66"
  }},
  {{
    "Name of fact": "Type",
    "Description of fact": " of casualty classifies the casualties as a civilian or non-civilian (e.g., military personnel are non-civilians).",
    "Common examples": "Civilian, Non-civilian"
  }},
  {{
    "Name of fact": "Actor",
    "Description of fact": " identifies the actors responsible for the incident, such as rebel groups.",
    "Common examples": "Leonardo Dicaprio, Brat Pit, goverment, etc."
  }},
]
END OF OUTPUT EXAMPLES

Topics should not be related among each other. Output {number_of_facts} facts in standard JSON form.
"""




# Auxilary functions

def extract_json(text):
    """
    Extracts a JSON array from a given text string and parses it into a Python object.

    Parameters:
    - text (str): The input string containing a JSON array along with additional text.

    Returns:
    - list: The extracted and parsed JSON array as a Python list.
    - None: If no JSON array is found or if parsing fails.
    """
    # Match JSON array within the text
    json_match = re.search(r'\[\s*{.*?}\s*\]', text, re.DOTALL)

    if json_match:
        json_string = json_match.group(0)  # Extract the matched JSON string
        try:
            # Parse the JSON data
            data = json.loads(json_string)
            return data
        except json.JSONDecodeError as e:
            print("Failed to decode JSON:", e)
            return None
    else:
        print("No JSON data found")
        return None

# Determine the path to the Fakes700 directory
parent_directory = os.path.dirname(os.getcwd())
fakes_directory = os.path.join(parent_directory, 'ACL2019_Data', 'Fakes700')

# Function to read and load contents of text files
def load_articles(file_paths):
    articles = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            articles.append(content)
    return articles

def generate_markdown_report(df):
    """This function takes the pandas DataFrame containing manipulated articles and generates a markdown report."""
    report = "# Report\n\n"
    
    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        report += f"## Row {index}\n\n"
        report += f"**Index:** {row['index']}\n\n"
        report += f"### Changed Article:\n\n{row['Changed_article']}\n\n"
        report += f"### Changed JSON:\n\n```json\n{row['Changed_json']}\n```\n\n"
        report += f"### Original Article:\n\n{row['Original_article']}\n\n"
        report += f"### Original JSON:\n\n```json\n{row['Original_json']}\n```\n\n"
        report += f"### Topics Changed:\n\n{', '.join(row['topics_changed'])}\n\n"
        report += f"### Topics:\n\n{row['topics']}\n\n"
        report += "---\n\n"
    
    return report

def security_save(df):
    """Save the DataFrame to a CSV file for security purposes."""
    import time

    current_time = time.strftime("%Y%m%d-%H%M%S")
    string = 'archive\manipulated_articles_{}.csv'.format(current_time)
    df.to_csv(string, index=False)






# Main function

def generate_fake_news(list_of_news, number_of_news, number_of_facts, number_of_facts_changed, testing=False, print_comments=False, strict=False):
  """
    Generate manipulated fake news articles by changing specific facts in the provided list of articles.

    This function samples a specified number of news articles from the input list, processes each article, and modifies a set number of facts within the article based on either pre-defined or model-generated topics. It returns the manipulated articles with labeled changes for further analysis.

    Parameters
    ----------
    list_of_news : pandas.DataFrame
        A list containing news articles or text in general.
    
    number_of_news : int
        The number of articles to sample and manipulate from the provided list of news.
    
    number_of_facts : int
        The number of facts to consider while processing each news article.
    
    number_of_facts_changed : int
        The number of facts to modify in each article to create fake news.
    
    testing : bool, optional, default=False
        If True, the function will use pre-defined testing topics for fact manipulation instead of generating them using the model.
    
    print_comments : bool, optional, default=False
        If True, intermediate steps and responses from the model will be printed for debugging or analysis purposes.
    
    strict : bool, optional, default=False
        If True, the function applies a stricter manipulation approach where facts are more aggressively changed.

    Returns
    -------
    results : pandas.DataFrame
        A DataFrame containing the manipulated articles along with the topics used for fact manipulation. The DataFrame includes both the original article and the modified version.

    Process
    -------
    1. Sample the specified number of articles from the input list.
    2. For each article, either use pre-defined topics (if testing is True) or generate topics using the "Llama3.1:8B" model.
    3. Extract and change the articles based on the generated or pre-defined topics using either a strict or non-strict method.
    4. Append the modified articles and topics to the result DataFrame.
    5. Return the final DataFrame containing all manipulated articles and their respective topics.

    Example
    -------
    >>> list_of_news = ['Article 1', 'Article 2', 'Article 3']
    >>> results = generate_fake_news(list_of_news, 2, 3, 1, testing=True, print_comments=True, strict=False)
    >>> print(results)
    """
  

  articles = random.sample(list_of_news, number_of_news)



  # Initialize an empty list to store results
  results = pd.DataFrame({})

  # Iterate over the articles and process each one
  for article in articles:

      if testing: # If we are youst testing the code, then use this pre-defined list of topics.
        topics = TESTING_TOPICS
      else:
        response = ollama.chat(model="Llama3.1:8B", messages=[
            {
                'role': 'user',
                'content': prompt.format(article=article, number_of_facts=number_of_facts),
                'temperature': 0.2,
            }
        ])

        output = response['message']['content']
        topics = extract_json(output)
      
        if print_comments: 
          print(output)


      if topics == None:
        print("Failed to extract topics from the response")
      else:
        print(topics)

        # Import the module for article manipulation
        sys.path.append(os.path.join(os.getcwd(), 'Task2'))
        import General_fact_extraction_and_manipulation as generating

        # Process the article and topics
        if strict:
          result = generating.extract_and_change_articles_with_labeling_added_strict(
            pd.DataFrame({'article_content': [article]}),
            topics,
            'llama3.1:8b',
            number_of_changed_facts = number_of_facts_changed,
            type_of_generation="one_by_one",
            print_comments=print_comments,
            testing=testing,
            change_of_article="paraphrase_aggressive",

        )
        else:
          result = generating.extract_and_change_articles_with_labeling_added(
            pd.DataFrame({'article_content': [article]}),
            topics,
            'llama3.1:8b',
            number_of_changed_facts = number_of_facts_changed,
            type_of_generation="one_by_one",
            print_comments=print_comments,
            testing=testing,
            change_of_article="paraphrase_aggressive",

        )
        # Extract the original article and topics from the DataFrame
        # Add the generated article and topics to the results DataFrame
        result['topics'] = json.dumps(topics , indent=4)
        results = pd.concat([result, results], ignore_index=True)

  # Output the list of all results
  security_save(results)
  return results


def generate_fake_news_repeated(list_of_news, number_of_news, number_of_facts, number_of_facts_changed, testing=False, print_comments=False, strict=False, max_number_of_trials=3):
  """
    Generate manipulated fake news articles by changing specific facts in the provided list of articles.

    This function samples a specified number of news articles from the input list, processes each article, and modifies a set number of facts within the article based on either pre-defined or model-generated topics. It returns the manipulated articles with labeled changes for further analysis.

    Parameters
    ----------
    list_of_news : pandas.DataFrame
        A list containing news articles or text in general.
    
    number_of_news : int
        The number of articles to sample and manipulate from the provided list of news.
    
    number_of_facts : int
        The number of facts to consider while processing each news article.
    
    number_of_facts_changed : int
        The number of facts to modify in each article to create fake news.
    
    testing : bool, optional, default=False
        If True, the function will use pre-defined testing topics for fact manipulation instead of generating them using the model.
    
    print_comments : bool, optional, default=False
        If True, intermediate steps and responses from the model will be printed for debugging or analysis purposes.
    
    strict : bool, optional, default=False
        If True, the function applies a stricter manipulation approach where facts are more aggressively changed.

    Returns
    -------
    results : pandas.DataFrame
        A DataFrame containing the manipulated articles along with the topics used for fact manipulation. The DataFrame includes both the original article and the modified version.

    Process
    -------
    1. Sample the specified number of articles from the input list.
    2. For each article, either use pre-defined topics (if testing is True) or generate topics using the "Llama3.1:8B" model.
    3. Extract and change the articles based on the generated or pre-defined topics using either a strict or non-strict method.
    4. Append the modified articles and topics to the result DataFrame.
    5. Return the final DataFrame containing all manipulated articles and their respective topics.

    Example
    -------
    >>> list_of_news = ['Article 1', 'Article 2', 'Article 3']
    >>> results = generate_fake_news(list_of_news, 2, 3, 1, testing=True, print_comments=True, strict=False)
    >>> print(results)
  """
  

  articles = random.sample(list_of_news, number_of_news)



  # Initialize an empty list to store results
  results = pd.DataFrame({})

  # Iterate over the articles and process each one
  for article in articles:
    generation_succeeded = False
    counter = 0
    while not generation_succeeded and counter <= max_number_of_trials:
      counter += 1
      if testing: # If we are youst testing the code, then use this pre-defined list of topics.
        topics = TESTING_TOPICS
      else:
        response = ollama.chat(model="Llama3.1:8B", messages=[
            {
                'role': 'user',
                'content': prompt.format(article=article, number_of_facts=number_of_facts),
                'temperature': 0.2,
            }
        ])

        output = response['message']['content']
        topics = extract_json(output)
      
        if print_comments: 
          print(output)


      if topics == None:
        print("Failed to extract topics from the response")
      else:
        print(topics)

        # Import the module for article manipulation
        sys.path.append(os.path.join(os.getcwd(), 'Task2'))
        import General_fact_extraction_and_manipulation as generating

        # Process the article and topics
        if strict:
          result = generating.extract_and_change_articles_with_labeling_added_strict(
            pd.DataFrame({'article_content': [article]}),
            topics,
            'llama3.1:8b',
            number_of_changed_facts = number_of_facts_changed,
            type_of_generation="one_by_one",
            print_comments=print_comments,
            testing=testing,
            change_of_article="paraphrase_aggressive",

        )
        else:
          result = generating.extract_and_change_articles_with_labeling_added_repeat(
            pd.DataFrame({'article_content': [article]}),
            topics,
            'llama3.1:8b',
            number_of_changed_facts = number_of_facts_changed,
            type_of_generation="one_by_one",
            print_comments=print_comments,
            testing=testing,
            change_of_article="paraphrase_aggressive",

        )
        # Extract the original article and topics from the DataFrame
        # Add the generated article and topics to the results DataFrame
        result.dropna()
        if result.empty:
          pass
        else:
          result['topics'] = json.dumps(topics , indent=4)
          results = pd.concat([result, results], ignore_index=True)
          generation_succeeded = True

  # Output the list of all results
  security_save(results)
  return results



def generate_fake_news_repeated_indices_added(list_of_news, number_of_news, number_of_facts, number_of_facts_changed, testing=False, print_comments=False, strict=False, max_number_of_trials=3, articles_ids=None):
  """
    Generate manipulated fake news articles by changing specific facts in the provided list of articles.

    This function samples a specified number of news articles from the input list, processes each article, and modifies a set number of facts within the article based on either pre-defined or model-generated topics. It returns the manipulated articles with labeled changes for further analysis.

    Parameters
    ----------
    list_of_news : pandas.DataFrame
        A list containing news articles or text in general.
    
    number_of_news : int
        The number of articles to sample and manipulate from the provided list of news.
    
    number_of_facts : int
        The number of facts to consider while processing each news article.
    
    number_of_facts_changed : int
        The number of facts to modify in each article to create fake news.
    
    testing : bool, optional, default=False
        If True, the function will use pre-defined testing topics for fact manipulation instead of generating them using the model.
    
    print_comments : bool, optional, default=False
        If True, intermediate steps and responses from the model will be printed for debugging or analysis purposes.
    
    strict : bool, optional, default=False
        If True, the function applies a stricter manipulation approach where facts are more aggressively changed.

    Returns
    -------
    results : pandas.DataFrame
        A DataFrame containing the manipulated articles along with the topics used for fact manipulation. The DataFrame includes both the original article and the modified version.

    Process
    -------
    1. Sample the specified number of articles from the input list.
    2. For each article, either use pre-defined topics (if testing is True) or generate topics using the "Llama3.1:8B" model.
    3. Extract and change the articles based on the generated or pre-defined topics using either a strict or non-strict method.
    4. Append the modified articles and topics to the result DataFrame.
    5. Return the final DataFrame containing all manipulated articles and their respective topics.

    Example
    -------
    >>> list_of_news = ['Article 1', 'Article 2', 'Article 3']
    >>> results = generate_fake_news(list_of_news, 2, 3, 1, testing=True, print_comments=True, strict=False)
    >>> print(results)
  """

  articles = random.sample(list_of_news, number_of_news)

  if  bool(articles_ids):
     articles_ids = [i for i in range(len(articles))]

  # Initialize an empty list to store results
  results = pd.DataFrame({})

  # Iterate over the articles and process each one
  for article, article_id in zip(articles, articles_ids):
    generation_succeeded = False
    counter = 0
    while not generation_succeeded and counter <= max_number_of_trials:
      counter += 1
      if testing: # If we are youst testing the code, then use this pre-defined list of topics.
        topics = TESTING_TOPICS
      else:
        response = ollama.chat(model="Llama3.1:8B", messages=[
            {
                'role': 'user',
                'content': prompt.format(article=article, number_of_facts=number_of_facts),
                'temperature': 0.2,
            }
        ])

        output = response['message']['content']
        topics = extract_json(output)
      
        if print_comments: 
          print(output)


      if topics == None:
        print("Failed to extract topics from the response")
      else:
        print(topics)

        # Import the module for article manipulation
        sys.path.append(os.path.join(os.getcwd(), 'Task2'))
        import General_fact_extraction_and_manipulation as generating

        # Process the article and topics
        if strict:
          result = generating.extract_and_change_articles_with_labeling_added_strict(
            pd.DataFrame({'article_content': [article]}),
            topics,
            'llama3.1:8b',
            number_of_changed_facts = number_of_facts_changed,
            type_of_generation="one_by_one",
            print_comments=print_comments,
            testing=testing,
            change_of_article="paraphrase_aggressive",

        )
        else:
          result = generating.extract_and_change_articles_with_labeling_added_repeat(
            pd.DataFrame({'article_content': [article]}),
            topics,
            'llama3.1:8b',
            number_of_changed_facts = number_of_facts_changed,
            type_of_generation="one_by_one",
            print_comments=print_comments,
            testing=testing,
            change_of_article="paraphrase_aggressive",

        )
        # Extract the original article and topics from the DataFrame
        # Add the generated article and topics to the results DataFrame
        result.dropna()
        if result.empty:
          pass
        else:
          result['topics'] = json.dumps(topics , indent=4)
          result['id'] = article_id
          results = pd.concat([result, results], ignore_index=True)
          
          generation_succeeded = True
  security_save(results)
  # Output the list of all results
  return results



