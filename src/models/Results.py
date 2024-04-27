import pandas as pd
import os
from src.data.TFQuestionnairesDataset import TFQuestionnairesDataset

class Results:
    # ------------
    # Constants
    # ------------
    PREDICTION_COLUMNS = ["QUESTIONNAIRE_ID", "GROUND_TRUTH_JSON", "PREDICTED_JSON", "REPORTED_EXCEPTION", 
                      "RESPONSE_TIME", "PROMPT_TOKENS", "COMPLETITION_TOKENS", "TOTAL_TOKENS",
                      "CONVERSION_ERROR"]
    QST_BLEU_COLUMNS = ["QUESTION", "BLEU_SCORE"]
    ASW_BLEU_COLUMNS = ["ANSWER", "BLEU_SCORE"]
    PREDICTIONS_FILENAME = "predictions.pkl"
    LOG_FILENAME = "log.txt"


    # ------------
    # Constructor
    # ------------
    def __init__(self):
        self.experiment_id = None
        self.questionnaire_id = None
        self.predictions = pd.DataFrame(columns=self.PREDICTION_COLUMNS)
        self.log = pd.DataFrame()
        self.generated_question_number = 0
        self.question_number_deviation = 0
        self.avg_generated_answer_number = 0
        self.avg_answer_number_deviation = 0
        self.question_bleu_scores = pd.DataFrame(columns=self.QST_BLEU_COLUMNS)
        self.answer_bleu_scores = pd.DataFrame(columns=self.ASW_BLEU_COLUMNS)

    
    # ------------
    # Methos
    # ------------
    def load_data(self, project_root, experiment_id):
        result_dir_path = os.path.join(project_root, "models", experiment_id)

        predictions_path = os.path.join(result_dir_path, self.PREDICTIONS_FILENAME)
        self.predictions = pd.read_pickle(predictions_path, encoding='latin1')

        log_path = os.path.join(result_dir_path, self.LOG_FILENAME)
        with open(log_path, 'r') as file:
            log_data = file.read()
        self.log = log_data


    def set_questionnaire_id(self, questionnaire_id):
        self.questionnaire_id = questionnaire_id


    def clear(self):
        self.questionnaire_id = None
        self.generated_question_number = 0
        self.question_number_deviation = 0
        self.avg_generated_answer_number = 0
        self.avg_answer_number_deviation = 0
        self.question_bleu_scores = pd.DataFrame(columns=self.QST_BLEU_COLUMNS)
        self.answer_bleu_scores = pd.DataFrame(columns=self.ASW_BLEU_COLUMNS)


    def compute_deviations(self):
        ground_truth = TFQuestionnairesDataset.from_json(self.predictions["GROUND_TRUTH_JSON"])
        predicted = TFQuestionnairesDataset.from_json(self.predictions["PREDICTED_JSON"])

        self.generated_question_number = len(predicted.questions)
        ground_truth_question_number = len(ground_truth.questions)

        self.question_number_deviation = self.generated_question_number - ground_truth_question_number

        # TODO: compute the average number of answers per question

    