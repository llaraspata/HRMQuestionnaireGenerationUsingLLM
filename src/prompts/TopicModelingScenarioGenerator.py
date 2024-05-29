from src.prompts.TopicModelingPromptGenerator import TopicModelingPromptGenerator

class PredictionScenarioGenerator:

    def __init__(self):
        self.system_prompt_generator = TopicModelingPromptGenerator("system")
        self.user_prompt_generator = TopicModelingPromptGenerator("user")
        self.assistant_prompt_generator = TopicModelingPromptGenerator("assistant")


    def generate_scenario(self, question):
        """
            Generate prompts for a given scenario, incorporating k-shot variations.

            Parameters:
                - question (str): The question whose topic will be extracted.

            Returns:
                - system prompt (str): Prompt for the system role.
                - user prompt (str): Prompt for the test user.
        """
        system_prompt = self.system_prompt_generator.generate_prompt()

        user_prompt = self.user_prompt_generator.generate_prompt(question)

        return system_prompt, user_prompt


    