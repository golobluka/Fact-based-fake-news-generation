
import os


#Import constants
#___________________________________________________________________________________________

current_path = os.getcwd()  # Get the current working directory
parent_directory = os.path.dirname(os.path.dirname(current_path))
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