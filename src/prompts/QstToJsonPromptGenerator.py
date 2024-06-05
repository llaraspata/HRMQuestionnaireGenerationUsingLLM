"""
This class is used for the generation of prompts w.r.t. to roles for the Topic Modeling task.
"""
from src.prompts.qst_to_json_prompts import QstToJsonSystemPrompt as SystemPrompt
from src.prompts.qst_to_json_prompts import QstToJsonUserPrompt as UserPrompt
from src.prompts.qst_to_json_prompts import QstToJsonAssistantPrompt as AssistantPrompt

class QstToJsonPromptGenerator:
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
            1. System role: Defines the task the language model has to carry out.
            2. User role: Builds the user prompt the questionnaire generation request.
        """

        if self.role == "system":
            self._generate_system_prompt()

        elif self.role == "user":
            self._generate_user_prompt(question)

        elif self.role == "assistant":
            self._generate_assistant_prompt()

        elif self.role == "sample_user":
            self._generate_demo_user_prompt()

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
        self.prompt += SystemPrompt.QUESTION_TYPES_DESCRIPTION
        
        self.prompt += SystemPrompt.FURTHER_SPECIFICATIONS


    def _generate_user_prompt(self, questionnaire):
        """
            Generate the user's prompt.

            Parameters:
                - questionnaire (str): The questionnaire text to be converted.

            Returns:
                - str: The formatted prompt.
        """
        self.prompt = UserPrompt.STANDARD_USER % (questionnaire)

    
    def _generate_demo_user_prompt(self):
        """
            Generate the demo user's prompt.
            The questionnaire to be converted is a fixed one for semplicity.
            
            Returns:
                - str: The formatted prompt.
        """
        self.prompt = UserPrompt.STANDARD_USER % (UserPrompt.SAMPLE_QUESTIONNAIRE_TXT)

    
    def _generate_assistant_prompt(self):
        """
            Generate the assistant prompt.

            Returns:
                - str: The formatted prompt.
        """
        self.prompt = AssistantPrompt.SAMPLE_CONVERTED_JSON