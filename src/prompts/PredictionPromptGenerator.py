"""
This class is used for the generation of prompts w.r.t. to roles for the Prediction task.
"""
from src.prompts.PredictionSystemPrompt import PredictionSystemPrompt as SystemPrompt
from src.prompts.PredictionUserPrompt import PredictionUserPrompt as UserPrompt
from src.prompts.PredictionAssistantPrompt import PredictionAssistantPrompt as AssistantPrompt

class PredictionPromptGenerator:
    # ------------
    # Constructor
    # ------------
    def __init__(self, role):
        self.role = role
        self.prompt = ""


    # ------------
    # Prompt generation
    # ------------
    def generate_prompt(self,
                        has_full_params=True, topic="", question_number=5,
                        question_types_data=[],
                        json=""):
        """
            Generate a prompt based on the specified role.

            Parameters:
            - has_full_params (bool): Whether the prompts includes all the questionnaire parameters.
            - topic (str): The questionnaire topic.
            - question_nuber (int): The number of questions the questionnaire must to have.
            - question_types_data (DataFrame): The possible types of questions.
            - json (str): The generated questionnaire.

            Returns:
            - str: The generated prompt.

            Roles:
            1. System role: Defines the task the language model has to carry out. If specified, it's possible to specify all the needed questionnaire's paramters.
            2. User role: Builds the user prompt the questionnaire generation request. If specified, it's possible to specify all the needed questionnaire's paramters.
            3. Assistant role: Used only for few-shot prompt scenarios, it formats the assistant's answer using a JSON format.
        """

        if self.role == "system":
            self._generate_system_prompt(has_full_params, question_types_data)

        elif self.role == "user":
            self._generate_user_prompt(has_full_params, topic, question_number)

        elif self.role == "assistant":
            self._generate_assistant_prompt(json)

        return self.prompt


    def _generate_system_prompt(self, has_full_params, question_types_data):
        """
            Generate the system's prompt.

            Parameters:
                - has_full_params (bool): Whether the prompts includes all the questionnaire parameters.
                - question_types_data (DataFrame): The possible types of questions.

            Returns:
                - str: The formatted prompt.
        """
        self.prompt = SystemPrompt.ROLE_DEFINITION
        
        if has_full_params:
            self.prompt += SystemPrompt.INPUT_FORMAT_DEFINITION_WITH_ALL_PARAMS
        else:
            self.prompt += SystemPrompt.INPUT_FORMAT_DEFINITION_WITH_ONLY_TOPIC

        self.prompt += SystemPrompt.ERROR_MESSAGE
        self.prompt += SystemPrompt.TASK_DEFINITION
        self.prompt += SystemPrompt.OUTPUT_FORMAT_DEFINITION

        self.prompt += SystemPrompt.QUESTION_TYPES_DESCRIPTION % (self._format_question_types(question_types_data))
        self.prompt += SystemPrompt.STYLE_COMMAND


    def _generate_user_prompt(self, has_full_params, topic, question_nuber):
        """
            Generate the user's prompt.

            Parameters:
                - has_full_params (bool): Whether the prompts includes all the questionnaire parameters.
                - topic (str): The questionnaire topic.
                - question_nuber (int): The number of questions the questionnaire must to have.

            Returns:
                - str: The formatted prompt.
        """
        if has_full_params:
            self.prompt = UserPrompt.STANDARD_USER_WITH_ALL_PARAMS % (topic, question_nuber)
        else:
            self.prompt = UserPrompt.STANDARD_USER_WITH_ONLY_TOPIC % (topic)


    def _generate_assistant_prompt(self, json):
        """
            Generate the assistant's prompt.

            Parameters:
                - json (str): The JSON string containing the questionnaire.

            Returns:
                - str: The formatted prompt.
        """
        self.prompt = AssistantPrompt.RESPONSE % (json)


    def _format_question_types(self, df):
        """
            Format the question types DataFrame into a string.

            Parameters:
                - df (DataFrame): The DataFrame containing the question types.

            Returns:
                - str: The formatted string.
        """
        result = ""

        for _, row in df.iterrows():
            result += SystemPrompt.SINGLE_QUESTION_TYPE_DESCRIPTION % (row["ID"], row["DESCRIPTION"])

        return result
