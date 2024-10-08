"""
This module contains all classes need to build prompts for the Prediction task.
"""

class PredictionSystemPrompt:
    ROLE_DEFINITION = """You are a Questionnaire Generator in the Human Resource Management field."""

    INPUT_FORMAT_DEFINITION_WITH_ALL_PARAMS = """\nThe user will ask you to generate a questionnaire specifying the topic and the number of questions."""
    INPUT_FORMAT_DEFINITION_WITH_ONLY_TOPIC = """\nThe user will ask you to generate a questionnaire about a specified topic."""

    ERROR_MESSAGE = """\nIf the user does not specify a valid topic, reply with "Sorry I cant help you"."""
    TASK_DEFINITION = """\nIf the topic is valid, reply with only a JSON, which must respect the following format:"""
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

    QUESTION_TYPES_DESCRIPTION = """\nThe admitted question's types are the following: %s"""
    SINGLE_QUESTION_TYPE_DESCRIPTION = """\n  - ID: %d, DESCRIPTION: %s"""

    STYLE_COMMAND = """\nBe creative and vary the syntax of your questions to enhance user engagement. Reply only with the JSON."""


class PredictionUserPrompt:
    STANDARD_USER_WITH_ALL_PARAMS = """Generate me a questionnaire on %s with %d questions"""
    STANDARD_USER_WITH_ONLY_TOPIC = """Generate me a questionnaire on %s"""


class PredictionAssistantPrompt:
    RESPONSE = """%s"""