

# Constants for list of changes

NAME_OF_CASUALTY = "Name of casualty or group"
NAME_OF_CASUALTY_DESCRIPTION = "Name of casualty or group is the name of the casualties or the name of the group associated with the casualties. It may also include the number of casualties."
NAME_OF_CASUALTY_QUESTION = """Name of casualty or group is the name of the casualties or the name of the group associated with the casualties. It may also include the number of casualties. Is the "Name of casualty or group" in the article approximately coherent with this description: {}? If the description comes from the article, output "The answer is true" and otherwise output "The answer is false". In addition to "The answer is true" or "The answer is false" label provide short explanation."""

GENDER_OR_AGE_GROUP = "Gender or age group"
GENDER_OR_AGE_GROUP_DESCRIPTION = "Gender or age group indicates if the casualty is male or female, or specifies their age group."
GENDER_OR_AGE_GROUP_QUESTION = """Gender or age group indicates if the casualty is male or female, or specifies their age group. Is the "Gender or age group" of the casualty in the article approximately coherent with this description: {}? If the description comes from the article, output "The answer is true" and otherwise output "The answer is false". In addition to "The answer is true" or "The answer is false" label provide short explanation."""

CAUSE_OF_DEATH = "Cause of death"
CAUSE_OF_DEATH_DESCRIPTION = "Cause of death is the weapon used in the attack (e.g., shooting, shelling, chemical weapons, etc.)."
CAUSE_OF_DEATH_QUESTION = """Cause of death is the weapon used in the attack (e.g., shooting, shelling, chemical weapons, etc.). Is the "Cause of death" in the article approximately coherent with this description: {}? If the description comes from the article, output "The answer is true" and otherwise output "The answer is false". In addition to "The answer is true" or "The answer is false" label provide short explanation."""

TYPE = "Type"
TYPE_DESCRIPTION = "Type is the information if the casualty is civilian or non-civilian."
TYPE_QUESTION = """Type is the information if the casualty is civilian or non-civilian. Is the "Type" (civilian or non-civilian) in the article approximately coherent with this description: {}? If the description comes from the article, output "The answer is true" and otherwise output "The answer is false". In addition to "The answer is true" or "The answer is false" label provide short explanation."""

ACTOR = "Actor"
ACTOR_DESCRIPTION = "The actor is the person or group responsible for the attack."
ACTOR_QUESTION = """The actor is the person or group responsible for the attack. Is the "Actor" (group responsible for the attack) in the article approximately coherent with this description: {}? If the description comes from the article, output "The answer is true" and otherwise output "The answer is false". In addition to "The answer is true" or "The answer is false" label provide short explanation."""

PLACE_OF_DEATH = "Place of death"
PLACE_OF_DEATH_DESCRIPTION = "Place of death refers to the cities or areas where the attacks happened."
PLACE_OF_DEATH_QUESTION = """Place of death refers to the cities or areas where the attacks happened. Is the "Place of death" in the article approximately coherent with this description: {}? If the description comes from the article, output "The answer is true" and otherwise output "The answer is false". In addition to "The answer is true" or "The answer is false" label provide short explanation."""

DATE_OF_DEATH = "Date of death"
DATE_OF_DEATH_DESCRIPTION = "The date of death refers to when the attack happened in the article."
DATE_OF_DEATH_QUESTION = """The date of death refers to when the attack happened in the article. Is the "Date of death" in the article approximately coherent with this description: {}? If the description comes from the article, output "The answer is true" and otherwise output "The answer is false". In addition to "The answer is true" or "The answer is false" label provide short explanation."""














# Constants for fake article generation

# Constants for CHANGES3
CHANGES3_ACTOR = 'The actor is the person or group responsible for the attack. You need to choose different facts for the value "Actor". Changed facts must bear different meaning. Common actors of Syrian war are: rebel groups, Russian forces, ISIS, the Syrian army, USA army, etc.'
CHANGES3_CAUSE_OF_DEATH = 'Cause of death is the weapon used in the attack. You need to choose different facts for the value "Cause of death". Changed facts must bear different meaning. Examples are shooting, shelling, chemical weapons, explosions, etc.'
CHANGES3_DATE_OF_DEATH = 'The date of death refers to time where the attack happened in the article. You need to choose different dates for the value "Date of death". Changed facts must bear different meaning.'
CHANGES3_PLACE_OF_DEATH = 'Place of death refers to the cities or areas where the attacks happened. You need to choose different facts for the value "Place of death". Changed facts must bear different meaning. Common places are Aleppo, Damascus, Homs, Idlib, Hasaka, Deir ez-Zor, Daraa, Qamishli or Tartus.'
CHANGES3_NAME_OF_CASUALTY_OR_GROUP = "Name of casualty or group is the name of the casualty or the name of the group associated with the casualty. You need to choose different facts for the value \"Name of casualty or group\". Changed facts must bear different meaning. Common examples are: Civilians (General category encompassing men, women, and children), Syrian Army (SAA), Free Syrian Army (FSA), National Liberation Front (NLF), Hay'at Tahrir al-Sham (HTS), People's Protection Units (YPG), Syrian Democratic Forces (SDF), Islamic State (ISIS), Hezbollah, Russian Forces, The White Helmets (Syrian Civil Defence), Kurdish Female Fighters (YPJ), Foreign Fighters (joining various factions), Al-Nusra Front Commanders, Civilians in Refugee Camps."

# Constants for CHANGES_AGGRESSIVE
CHANGES_AGGRESSIVE_ACTOR = 'The actor is the person or group responsible for the attack. You need to choose totally different facts for the values of "Actor". Changed values must bear different meaning. Common actors of Syrian war are: rebel groups, Russian forces, ISIS, the Syrian army, USA army, etc.'
CHANGES_AGGRESSIVE_CAUSE_OF_DEATH = 'Cause of death is the weapon used in the attack. You need to choose totally different facts for the values of "Cause of death". Changed values must bear different meaning. Examples are shooting, shelling, chemical weapons, explosions, etc.'
CHANGES_AGGRESSIVE_DATE_OF_DEATH = 'The date of death refers to time where the attack happened in the article. You need to choose totally different dates for the values of "Date of death". Changed values must bear different meaning. Use date that precedes the date for up to 1 year.'
CHANGES_AGGRESSIVE_PLACE_OF_DEATH = 'Place of death refers to the cities or areas where the attacks happened. You need to choose totally different facts for the values of "Place of death". Changed values must bear different meaning. Common places are Aleppo, Damascus, Homs, Idlib, Hasaka, Deir ez-Zor, Daraa, Qamishli or Tartus.'
CHANGES_AGGRESSIVE_NAME_OF_CASUALTY_OR_GROUP = "Name of casualty or group is the name of the casualty or the name of the group associated with the casualty. You need to choose totally different facts for the values of \"Name of casualty or group\". Changed values must bear different meaning. Common examples are: Civilians (General category encompassing men, women, and children), Syrian Army (SAA), Free Syrian Army (FSA), National Liberation Front (NLF), Hay'at Tahrir al-Sham (HTS), People's Protection Units (YPG), Syrian Democratic Forces (SDF), Islamic State (ISIS), Hezbollah, Russian Forces, The White Helmets (Syrian Civil Defence), Kurdish Female Fighters (YPJ), Foreign Fighters (joining various factions), Al-Nusra Front Commanders, Civilians in Refugee Camps."

# Constants for TOPICS
TOPICS_NAME_OF_CASUALTY_OR_GROUP = " represents the casualties names or the names of the groups associated with the casualties."
TOPICS_GENDER_OR_AGE_GROUP = " of casualty indicates if the casualties are male or female, or specify their age group (e.g., child, adult, senior)."
TOPICS_CAUSE_OF_DEATH = " specifies the weapons used by the aggressor (e.g., shooting, shelling, chemical weapons, etc.)"
TOPICS_TYPE = " of casualty classifies the casualties as a civilian or non-civilian (e.g., military personnel are non-civilians)."
TOPICS_ACTOR = " identifies the actors responsible for the incident, such as rebel groups, Russian forces, ISIS, the Syrian army, U.S. military, etc."
TOPICS_PLACE_OF_DEATH = " specifies the locations where the attacks occurred (e.g., Aleppo, Damascus, Homs, Idlib, Raqqa, Daraa, Deir ez-Zor, Qamishli, Palmyra, etc.)."
TOPICS_DATE_OF_DEATH = " provides the dates when the attacks occurred."











# General_fact_extraction_and_manipulation constants

GENERAL_NAME_OF_FACT = 'Name of fact'
GENERAL_DESCRIPTION_OF_FACT = 'Description of fact'
GENERAL_COMMON_EXAMPLES = 'Common examples'











#General_news_labeling constants

FACT1_NAME = 'Turnout'
FACT1_DESCRIPTION = 'The number of people who take part in election.'
FACT1_EXAMPLES = '1000, 10k'

FACT2_NAME = 'Gender or age group'
FACT2_DESCRIPTION = ' of casualty indicates if the casualties are male or female, or specify their age group .'
FACT2_EXAMPLES = 'Male, Female, Child, Adult, Senior'

FACT3_NAME = 'Cause of death'
FACT3_DESCRIPTION = ' specifies the weapons used by the aggressor (e.g., shooting, shelling, chemical weapons, etc.)'
FACT3_EXAMPLES = 'Shooting, Shelling, Chemical weapons'

FACT4_NAME = 'Type'
FACT4_DESCRIPTION = ' of casualty classifies the casualties as a civilian or non-civilian (e.g., military personnel are non-civilians).'
FACT4_EXAMPLES = 'Civilian, Non-civilian'

FACT5_NAME = 'Actor'
FACT5_DESCRIPTION = ' identifies the actors responsible for the incident, such as rebel groups, Russian forces, ISIS, the Syrian army, U.S. military, etc.'
FACT5_EXAMPLES = 'Rebel groups, Russian forces, ISIS'

FACT6_NAME = 'Place of death'
FACT6_DESCRIPTION = ' specifies the locations where the attacks occurred (e.g., Aleppo, Damascus, Homs, Idlib, Raqqa, Daraa, Deir ez-Zor, Qamishli, Palmyra, etc.).'
FACT6_EXAMPLES = 'Aleppo, Damascus, Homs'

FACT7_NAME = 'Date of death'
FACT7_DESCRIPTION = ' provides the dates when the attacks occurred.'
FACT7_EXAMPLES = '2021-01-01, 2022-06-15'





#Testing_responses



testing_dictionary = {
    'fact_extraction': [
        {
          "Name of casualty or group": "Displaced children",
          "Gender or age group": "Child",
          "Cause of death": "Suicide bombing by Daesh militants",
          "Type": "Civilian",
          "Actor": "Daesh militants",
          "Place of death": "Rajm Sleibi, Hasakeh province",
          "Date of death": "Not available"
        }],
    'article_generation': """After analyzing the article and the related facts extracted from it, I have made the following decisions:

    **Point 1:** The new value for "Cause of death" is Shelling by Daesh militants.
    
    This change gives a different meaning to the cause of death compared to the original value of Suicide bombing by Daesh militants.
    
    **Point 2:** Here is the new JSON file with the changed value for Cause of death:
    
    
    \{
        "Name of casualty or group": "Displaced children",
        "Gender or age group": "Child",
        "Cause of death": "Shelling by Daesh militants",
        "Type": "Civilian",
        "Actor": "Syrian government forces",
        "Place of death": "Rajm Sleibi, Hasakeh province",
        "Date of death": "Not available"
    \}

    
    **Point 3:** Here is the new article with the changed value for Cause of death:
    
    BEGINNING OF THE ARTICLE
    
    2 May 2017 At least 24 people were killed on Tuesday in a Daesh attack near a refugee camp on the Syrian side of the border with Iraq, a Britain-based war monitor said. The Syrian Observatory for Human Rights said militants sneaked into the village of Rajm Sleibi which houses the camp for the displaced people fleeing Daesh-held areas in Syria and Iraq. The village lies in Hasakeh province, a frontline that separates the area from Daesh-held places further south. "At least five suicide attackers blew themselves up outside and inside a camp for Iraqi refugees and displaced Syrians in Hasakeh province" Observatory chief Rami Abdel Rahman said. The dead included displaced children who died after being shelled by Daesh militants.
    
    Heavy clashes then erupted between the Daesh fighters and members of the US-backed Syrian Democratic Forces (SDF), an alliance of Kurdish and Arab fighters some of whose combatants were among the dead Abdel Rahman said. The SDF is dominated by the YPG which is the armed wing of the PYD, a Syrian affiliate of the PKK listed as a terrorist organisation by Turkey, the US and EU. Redur Khalil one of the spokespersons within SDF said the attack came a few hours after Daesh suicide bombers dressed in civilian clothes entered the town of Shaddadeh and engaged SDF forces triggering clashes.
    
    The Syrian government forces were quick to respond to the attack, with officials confirming that they had sent troops to the area to assist in the counter-attack against the terrorist group. Meanwhile, Human Rights Watch on Monday accused Syrian government forces of likely dropping bombs containing nerve agents at least three times elsewhere in the country before an April 4 attack that killed dozens of people and sparked a retaliatory US strike.
    
    Bashar al Assad's forces are also stepping up chlorine gas attacks and have begun using surface-fired rockets filled with chlorine in fighting near Damascus, the US-based rights group said in a new report. "The government's use of nerve agents is a deadly escalation and part of a clear pattern" said Kenneth Roth, Human Rights Watch's executive director.
    
    The Organisation for the Prohibition of Chemical weapons, a global watchdog, has said sarin or a similar banned toxin was used in the April 4 strike. Human Rights Watch said that before the April 4 attack on Khan Shaykhun, government warplanes also appeared to have dropped nerve agents on eastern Hama on December 11 and 12, 2016, and northern Hama near Khan Shaykhun on March 30, 2017.
    
    Three suspected attacks in Hama Human Rights Watch said 64 people died from exposure to nerve agents after warplanes attacked territory controlled by Daesh in eastern Hama on December 11 and December 12. Activists and local residents provided names of the victims while Human Rights Watch interviewed four witnesses and two medical personnel about the alleged attacks.
    
    A third suspected nerve agent attack in northern Hama on March 30 caused no deaths but injured dozens of civilians and combatants according to residents and medical personnel the report said. The alleged attacks were systematic and in some cases directed against civilians which would meet the legal criteria to be characterised as war crimes, said the Syrian government forces.
    
    END OF THE ARTICLE
    
    **Point 4:** After reviewing the new article, I am satisfied that all occurrences of Cause of death have been changed to Shelling by Daesh militants, and the content from the original article has been preserved in a consistent way.
        """,
    'labeling_true': 'The answer is true.',
    'labeling_false': 'The answer is false'
 }

TESTING_TOPICS = [
          {
            "Name of fact": "Number of hostages",
            "Description of fact": "The total number of people being held by terrorists in Gaza.",
            "Common examples": "240"
          },
          {
            "Name of fact": "Status of release negotiations",
            "Description of fact": "Information about the progress and prospects of releasing the hostages.",
            "Common examples": "Ongoing, Uncertain, Possible"
          },
          {
            "Name of fact": "Humanitarian aid efforts",
            "Description of fact": "Efforts to increase the flow of aid into Gaza to alleviate suffering among the civilian population.",
            "Common examples": "Increase, Implement localized pauses, Discuss additional pauses"
          }
        ]









