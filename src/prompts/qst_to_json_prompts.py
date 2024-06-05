"""
This module contains all classes need to build prompts for the Topic Modeling task.
"""

class QstToJsonSystemPrompt:
    ROLE_DEFINITION = """You are a JSON converter."""

    INPUT_FORMAT_DEFINITION = """\nThe user will ask you to convert the questionnaire text into a JSON."""

    TASK_DEFINITION = """\nYou must reply with only the JSON which has the following structure: """
    OUTPUT_FORMAT_DEFINITION = """
        - The root of the JSON is an object that contains a single property 'data'.
        - The 'data' property is an object that contains a single property 'TF_QUESTIONNAIRES'.
        - 'TF_QUESTIONNAIRES' is an array of only one element, which represents a questionnaire. It has the following properties:
            - 'CODE': (string) the questionnaire's code.
            - 'NAME': (string) the questionnaire's name.
            - 'TYPE_ID': (int) which is equal to 3.
            - '_TF_QUESTIONS': An array of objects, each representing a question.
        - Each question in the '_TF_QUESTIONS' array has the following properties:
            - 'CODE': (string) the question's unique code.
            - 'NAME': (string) the question's content.
            - 'TYPE_ID': (int) the question's type.
            - 'DISPLAY_ORDER': (int) the question's display order.
            - '_TF_ANSWERS': An array of objects, each representing a possible answer to the question.
        - Each answer object in the '_TF_ANSWERS' array has a single property 'ANSWER', which is a string representing the content of the answer."""
    
    QUESTION_TYPES_DESCRIPTION = """The admited question types are:
        - ID: 1, DESCRIPTION: Use this type of question to choose one answer from a list.
        - ID: 2, DESCRIPTION: Use this type of question to choose one or more answer from a list.
        - ID: 3, DESCRIPTION: Use this type of question to rate something.
        - ID: 4, DESCRIPTION: Use this type of question to aquire feedback for open questions."""

    FURTHER_SPECIFICATIONS = """
The enumeration of questions can be broken, so fix it while converting it.
The '[ ]' does not stand for a multi-choise question, so use the meaning of the question to decide its type.
The '______________________" stands for an open question. 
When the question is open, then '_TF_ANSWERS' is an empty string."""


class QstToJsonUserPrompt:
    STANDARD_USER = """QUESTIONNAIRE TO CONVERT: \n%s"""

