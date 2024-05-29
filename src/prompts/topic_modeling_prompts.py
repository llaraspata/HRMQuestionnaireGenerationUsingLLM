"""
This module contains all classes need to build prompts for the Topic Modeling task.
"""

class TopicModelingSystemPrompt:
    ROLE_DEFINITION = """You are a Topic Modeler in the Human Resources Management field."""

    INPUT_FORMAT_DEFINITION = """\nThe user will ask you to extract the topic of a question."""

    TASK_DEFINITION = """\nYou must reply with only the topic of the question. """
    OUTPUT_FORMAT_DEFINITION = """Use at most 3 words to describe the topic of the question."""


class TopicModelingUserPrompt:
    STANDARD_USER = """Give me only the topic of the question '%s'."""

