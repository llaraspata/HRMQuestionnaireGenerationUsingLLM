from src.prompts.QstToJsonPromptGenerator import QstToJsonPromptGenerator

class QstToJsonScenarioGenerator:

    def __init__(self):
        self.system_prompt_generator = QstToJsonPromptGenerator("system")
        self.user_prompt_generator = QstToJsonPromptGenerator("user")
        self.demo_user_prompt_generator = QstToJsonPromptGenerator("user")
        self.assistant_prompt_generator = QstToJsonPromptGenerator("assistant")


    def generate_scenario(self, questionnaire):
        """
            Generate prompts for a given scenario.

            Parameters:
                - questionnaire (str): The questionnaire text to be converted to JSON.

            Returns:
                - system prompt (str): Prompt for the system role.
                - user prompt (str): Prompt for the test user.
        """
        system_prompt = self.system_prompt_generator.generate_prompt()

        user_prompt = self.user_prompt_generator.generate_prompt(questionnaire)
        demo_user_prompt = self.demo_user_prompt_generator.generate_prompt()
        assistant_prompt = self.assistant_prompt_generator.generate_prompt()

        return system_prompt, user_prompt, demo_user_prompt, assistant_prompt


    