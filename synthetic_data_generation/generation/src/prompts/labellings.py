
import sys
import os

current_path = os.getcwd()  # Get the current working directory
parent_directory = os.path.dirname(os.path.dirname(os.path.dirname(current_path)))
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