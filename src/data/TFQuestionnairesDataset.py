import json
import pandas as pd
import os
import itertools
import random

class TFQuestionnairesDataset:
    # ------------
    # Constants
    # ------------
    QUESTIONNAIRES_FILENAME = "TF_QST_QUESTIONNAIRES.csv"
    QUESTIONS_FILENAME = "TF_QST_QUESTIONS.csv"
    QUESTION_TYPES_FILENAME = "TF_QST_QUESTION_TYPES.csv"
    ANSWERS_FILENAME = "TF_QST_ANSWERS.csv"

    ESSENTIAL_COLUMNS_QUESTIONNAIRES = ["ID", "CODE", "NAME", "DESCRIPTION"]
    ESSENTIAL_COLUMNS_QUESTIONS = ["ID", "TYPE_ID", "QUESTIONNAIRE_ID", "CODE", "NAME", "DISPLAY_ORDER"]
    ESSENTIAL_COLUMNS_QUESTION_TYPES = ["ID", "CODE", "NAME", "DESCRIPTION"]
    ESSENTIAL_COLUMNS_ANSWERS = ["ID", "QUESTION_ID", "ANSWER"]

    QUESTIONNAIRES_PROMPT_COLUMNS = ["CODE", "NAME"]
    QUESTIONS_PROMPT_COLUMNS = ["CODE", "NAME", "TYPE_ID", "DISPLAY_ORDER"]
    ANSWERS_PROMPT_COLUMNS = ["ANSWER"]


    # ------------
    # Constructor
    # ------------
    def __init__(self):
        self.questionnaires = None
        self.questions = None
        self.question_types = None
        self.answers = None

    
    # ------------
    # Methos
    # ------------
    def load_data(self):
        project_root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

        data_dir_path = os.path.join(project_root, "data", "raw", "starter")

        questionnaires_path = os.path.join(data_dir_path, self.QUESTIONNAIRES_FILENAME)
        questions_path = os.path.join(data_dir_path, self.QUESTIONS_FILENAME)
        question_types_path = os.path.join(data_dir_path, self.QUESTION_TYPES_FILENAME)
        answer_path = os.path.join(data_dir_path, self.ANSWERS_FILENAME)

        self.questionnaires = pd.read_csv(questionnaires_path, encoding='latin1')
        self.questions = pd.read_csv(questions_path, encoding='latin1')
        self.question_types = pd.read_csv(question_types_path, encoding='latin1')
        self.answers = pd.read_csv(answer_path, encoding='latin1')


    def load_question_types(self):
        project_root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        data_dir_path = os.path.join(project_root, "data", "raw", "starter")
        question_types_path = os.path.join(data_dir_path, self.QUESTION_TYPES_FILENAME)
        self.question_types = pd.read_csv(question_types_path, encoding='latin1')

    
    def get_sample_questionnaire_data(self):
        sample_instance = TFQuestionnairesDataset()

        sample_instance.questionnaires = self.questionnaires.iloc[0]
        sample_instance.question_types = self.question_types
        sample_instance.questions = self.questions[self.questions["QUESTIONNAIRE_ID"] == sample_instance.questionnaires["ID"]]
        sample_instance.answers = self.answers[self.answers["QUESTION_ID"].isin(sample_instance.questions["ID"])]

        return sample_instance


    def get_question_types(self):
        return self.question_types[self.ESSENTIAL_COLUMNS_QUESTION_TYPES]
    

    def to_json(self, questionnaire_id):
        questionnaire = self.questionnaires[self.questionnaires["ID"] == questionnaire_id]
        questions = self.questions[self.questions["QUESTIONNAIRE_ID"] == questionnaire_id]
        answers = self.answers[self.answers["QUESTION_ID"].isin(questions["ID"])]

        questionnaire_json = questionnaire[self.QUESTIONNAIRES_PROMPT_COLUMNS].to_dict(orient="records")
        questions_json = questions[self.QUESTIONS_PROMPT_COLUMNS].to_dict(orient="records")
        answers_json = answers[self.ANSWERS_PROMPT_COLUMNS].to_dict(orient="records")

        data = {
            "data": {
                "TF_QUESTIONNAIRES": questionnaire_json,
            }
        }

        for question in questions_json:
            question["_TF_ANSWERS"] = [answer for answer in answers_json]
        
        data["data"]["TF_QUESTIONNAIRES"][0]["_TF_QUESTIONS"] = questions_json

        return json.dumps(data)
    

    def from_json(json_data):
        result = TFQuestionnairesDataset()
        data = json.loads(json_data)["data"]

        questionnaire = data["TF_QUESTIONNAIRES"]
        questions = questionnaire[0]["_TF_QUESTIONS"]
        answers = [question["_TF_ANSWERS"] for question in questions]
        answers_flat = list(itertools.chain.from_iterable(answers))

        result.questionnaires = pd.DataFrame(questionnaire)[result.QUESTIONNAIRES_PROMPT_COLUMNS]
        result.questions = pd.DataFrame(questions)[result.QUESTIONS_PROMPT_COLUMNS]
        result.answers = pd.DataFrame(answers_flat)[result.ANSWERS_PROMPT_COLUMNS]
        result.question_types = result.load_question_types()

        return result


    def get_questionnaire_topic(self, questionnaire_id):
        return self.questionnaires[self.questionnaires["ID"] == questionnaire_id]["TOPIC"]


    def get_questionnaire_question_type(self, questionnaire_id):
        return self.questions[self.questions["QUESTIONNAIRE_ID"] == questionnaire_id]["TYPE_ID"].values[0]
    

    def get_questionnaire_question_number(self, questionnaire_id):
        questions_ids = self.questions[self.questions["QUESTIONNAIRE_ID"] == questionnaire_id]["ID"]

        return questions_ids.count()
    

    def get_sample_questionnaire_id(self, sample_questionnaire_ids, current_questionnaire_id):
        """
            Gets a random questionnaire id from, excluding those in the sample questionnaire ids list.
        """
        n_questionnaires = len(self.questionnaires["ID"])
        random_index = current_questionnaire_id

        while random_index == current_questionnaire_id:
            random_index = random.randint(0, n_questionnaires - 1)

            if self.questionnaires["ID"][random_index] in sample_questionnaire_ids:
                random_index = current_questionnaire_id

        return self.questionnaires["ID"][random_index]