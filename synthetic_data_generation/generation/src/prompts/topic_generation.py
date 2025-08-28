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
