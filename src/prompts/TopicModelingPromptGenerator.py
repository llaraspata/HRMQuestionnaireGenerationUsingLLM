"""
This class is used for the generation of prompts w.r.t. to roles for the Topic Modeling task.
"""
from src.prompts.topic_modeling_prompts import TopicModelingSystemPrompt as SystemPrompt
from src.prompts.topic_modeling_prompts import TopicModelingUserPrompt as UserPrompt

class TopicModelingPromptGenerator:
    # ------------
    # Constructor
    # ------------
    def __init__(self, role):
        self.role = role
        self.prompt = ""


    # ------------
    # Prompt generation
    # ------------
    def generate_prompt(self, question=""):
        """
            Generate a prompt based on the specified role.

            Parameters:
            - question (str): The question whose topic will be extracted.

            Returns:
            - str: The generated prompt.

            Roles:
            1. System role: Defines the task the language model has to carry out. If specified, it's possible to specify all the needed questionnaire's paramters.
            2. User role: Builds the user prompt the questionnaire generation request. If specified, it's possible to specify all the needed questionnaire's paramters.
        """

        if self.role == "system":
            self._generate_system_prompt()

        elif self.role == "user":
            self._generate_user_prompt(question)

        return self.prompt


    def _generate_system_prompt(self):
        """
            Generate the system's prompt.

            Returns:
                - str: The formatted prompt.
        """
        self.prompt = SystemPrompt.ROLE_DEFINITION

        self.prompt += SystemPrompt.INPUT_FORMAT_DEFINITION
        self.prompt += SystemPrompt.TASK_DEFINITION

        self.prompt += SystemPrompt.OUTPUT_FORMAT_DEFINITION


    def _generate_user_prompt(self, question):
        """
            Generate the user's prompt.

            Parameters:
                - question (str): The question whose topic will be extracted.

            Returns:
                - str: The formatted prompt.
        """
        self.prompt = UserPrompt.STANDARD_USER % (question)