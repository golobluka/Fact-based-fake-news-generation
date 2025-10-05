import sys
import os
import random
import ollama
import json
import re
import pandas as pd
import General_news_labeling


from prompts.fact_extraction import  GENERAL_NAME_OF_FACT, GENERAL_DESCRIPTION_OF_FACT, GENERAL_COMMON_EXAMPLES, GENERAL_CHANGES, GENERAL_CHANGES_CHANGE_MEANING, GENERAL_CHANGES_FUNNY, GENERAL_TOPICS, GENERAL_INFORMATION, PROMPT_FOR_CORRECTING_MISTAKES_WHEN_CHANGING_FACTS, GENERAL_PROMPT_FACT_EXTRACTION, GENERAL_PROMPT_FOR_ONE_BY_ONE_FACT_EXTRACTION_CONCISE_VERSION, GENERAL_CHANGE_ONE_INFORMATION
from tool_functions.tool_functions import find_json, extract_last_article, extract_last_text, print_readable_dict, extract_fact_from_text, generate_aggressive_prompts, model_response
from generating_facts.different_types_of_generation import generate_facts_normal, generate_facts_one_by_one, generate_facts


#Directory for constants file
current_path = os.path.abspath(__file__) # Get the current working directory
directory_three_files_back = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_path))))
print(directory_three_files_back)
sys.path.append(directory_three_files_back)
from constants import testing_dictionary









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