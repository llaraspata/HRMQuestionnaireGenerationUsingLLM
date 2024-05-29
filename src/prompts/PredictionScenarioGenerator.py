from src.prompts.PredictionPromptGenerator import PredictionPromptGenerator

class PredictionScenarioGenerator:

    def __init__(self, experiment_config, dataset, full_log=True):
        self.experiment_config = experiment_config
        self.system_prompt_generator = PredictionPromptGenerator("system")
        self.user_prompt_generator = PredictionPromptGenerator("user")
        self.assistant_prompt_generator = PredictionPromptGenerator("assistant")
        self.full_log = full_log
        self.question_types = dataset.get_question_types()


    def generate_scenario(self, log_file, current_questionnaire_id, dataset):
        """
            Generate prompts for a given scenario, incorporating k-shot variations.

            Parameters:
                - log_file (str): The path to the log file.
                - current_questionnaire_id (int): The questionnaire identifier to be used as test.
                - dataset (TFQuestionnairesDataset): The dataset containing the questionnaires.

            Returns:
                - system prompt (str): Prompt for the system role.
                - sample user prompts (list<str>): List of prompts for the sample users role (used only in few-shots scenarios).
                - assistant prompts (list<str>): List of prompts for the assistants role (used only in few-shots scenarios).
                - user prompt (str): Prompt for the test user.
                - sample_questionnaires_ids (list<int>): List of the sample questionnaires identifiers.
        """
        system_prompt, sample_user_prompts, assistant_prompts, user_prompt, sample_questionnaires_ids = self.generate_k_shot_scenario(log_file,
                                                                                                                                      current_questionnaire_id, dataset)

        return system_prompt, sample_user_prompts, assistant_prompts, user_prompt, sample_questionnaires_ids

    def generate_k_shot_scenario(self, log_file, current_questionnaire_id, dataset):
        """
            Generate prompts for a k-shot scenario.

            Parameters:
                - log_file (str): The path to the log file.
                - current_questionnaire_id (int): The questionnaire identifier to be used as test.
                - dataset (TFQuestionnairesDataset): The dataset containing the questionnaires.

            Returns:
                - system prompt (str): Prompt for the system role.
                - sample user prompts (list<str>): List of prompts for the sample users role (used only in few-shots scenarios).
                - assistant prompts (list<str>): List of prompts for the assistants role (used only in few-shots scenarios).
                - user prompt (str): Prompt for the test user.
                - ground truth (dict): The ground truth for the test questionnaire.
                - sample_questionnaires_ids (list<int>): List of the sample questionnaires identifiers.

        """
        has_full_params = self.experiment_config["has_full_params"]
        k = self.experiment_config["k"]

        sample_questionnaires_ids = []
        all_sample_user_prompts = []
        all_assistant_prompts = []

        log_file.write("\n=================================================================")
        log_file.write(f"\n{k}-SHOT")
        if has_full_params:
            log_file.write("\nWith FULL questionnaire parameters")
        else:
            log_file.write("\nWith only the questionnaire TOPIC")
        log_file.write("\n=================================================================")


        # -------------------
        # SYSTEM PROMPT
        # -------------------
        system_prompt = self.system_prompt_generator.generate_prompt(has_full_params=has_full_params, question_types_data=self.question_types)

        # -------------------
        # SAMPLE USER AND ASSISTANT PROMPTS
        # -------------------
        for _ in range(k):
            # Get sample user's id
            sample_questionnaire_id = dataset.get_sample_questionnaire_id(sample_questionnaires_ids, current_questionnaire_id)
            sample_questionnaires_ids.append(sample_questionnaire_id)

            topic = dataset.get_questionnaire_topic(sample_questionnaire_id)
            question_number = dataset.get_questionnaire_question_number(sample_questionnaire_id)

            formatted_json = dataset.to_json(sample_questionnaire_id)

            if self.full_log:
                log_file.write("\n\n-------------------")
                log_file.write(f"\n[SAMPLE] QUESTIONNAIRE_ID: {current_questionnaire_id}")
                log_file.write(f"\n     - Topic: {topic}")
                log_file.write(f"\n     - Question number: {question_number}")

            # Build sample user and assistant prompts
            sample_user_prompt = self.user_prompt_generator.generate_prompt(has_full_params=has_full_params, topic=topic,
                                                                            question_number=question_number)

            assistant_prompt = self.assistant_prompt_generator.generate_prompt(json=formatted_json)

            all_sample_user_prompts.append([sample_user_prompt])
            all_assistant_prompts.append([assistant_prompt])

        # -------------------
        # TEST USER PROMPT
        # -------------------
        topic = dataset.get_questionnaire_topic(current_questionnaire_id)
        question_number = dataset.get_questionnaire_question_number(current_questionnaire_id)

        log_file.write("\n\n-------------------")
        log_file.write(f"\n[TEST] QUESTIONNAIRE_ID: {current_questionnaire_id}")

        if self.full_log:
            log_file.write(f"\n     - Topic: {topic}")
            log_file.write(f"\n     - Question number: {question_number}")

        user_prompt = self.user_prompt_generator.generate_prompt(has_full_params=has_full_params, topic=topic,
                                                                 question_number=question_number)

        return system_prompt, all_sample_user_prompts, all_assistant_prompts, user_prompt, sample_questionnaires_ids


    