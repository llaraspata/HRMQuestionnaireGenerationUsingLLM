import pandas as pd
import os

class TFQuestionnairesDataset:
    # ------------
    # Constants
    # ------------
    QUESTIONNAIRES_FILENAME = "TF_QST_QUESTIONNAIRES.csv"
    QUESTIONS_FILENAME = "TF_QST_QUESTIONS.csv"
    QUESTION_TYPES_FILENAME = "TF_QST_QUESTION_TYPES.csv"
    ANSWERS_FILENAME = "TF_QST_ANSWERS.csv"

    ESSENTIAL_COLUMNS_QUESTIONNAIRES = ["ID", "CODE", "NAME", "DESCRIPTION"]
    ESSENTIAL_COLUMNS_QUESTIONS = ["ID", "TYPE_ID", "QUESTIONNAIRE_ID", "CODE", "NAME"]
    ESSENTIAL_COLUMNS_QUESTION_TYPES = ["ID", "CODE", "NAME", "DESCRIPTION"]
    ESSENTIAL_COLUMNS_ANSWERS = ["ID", "QUESTION_ID", "ANSWER", "SCORE"]

    # ------------
    # Constructor
    # ------------
    def __init__(self):
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
