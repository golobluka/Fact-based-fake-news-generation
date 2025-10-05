import sys
import os
import random
import ollama
import json
import re
import pandas as pd

#Get prompts from prompt package
prompt_package_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
sys.path.append(prompt_package_directory)
from prompts.fact_extraction import GENERAL_TOPICS, GENERAL_INFORMATION, GENERAL_PROMPT_FOR_ONE_BY_ONE_FACT_EXTRACTION_CONCISE_VERSION


#Get axiliarry functions from tool functions package
prompt_package_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tool_functions")
sys.path.append(prompt_package_directory)
from tool_functions.tool_functions  import model_response, extract_fact_from_text, find_json






#Auxiliarry funcion
#______________________________________________

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






# Generate facts function: We have three different types.
#________________________________________________________---


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