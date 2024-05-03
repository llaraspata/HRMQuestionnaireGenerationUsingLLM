import pandas as pd
import os
from src.data.TFQuestionnairesDataset import TFQuestionnairesDataset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

class Results:
    # ------------
    # Constants
    # ------------
    PREDICTION_COLUMNS = ["QUESTIONNAIRE_ID", "GROUND_TRUTH_JSON", "PREDICTED_JSON", "REPORTED_EXCEPTION", 
                      "RESPONSE_TIME", "PROMPT_TOKENS", "COMPLETITION_TOKENS", "TOTAL_TOKENS",
                      "CONVERSION_ERROR"]
    
    STATISTICS_COLUMNS = ["QUESTIONNAIRE_ID", 
                          "CONVERSION_ERROR", "GENERATED_QUESTIONNAIRES", "QUESTIONS_WITH_MISSING_ANSWERS", 
                          "GENERATED_QUESTION_NUMBER", "QUESTION_NUMBER_DEVIATION", 
                          "AVERAGE_GENERATED_ANSWER_NUMBER", "AVERAGE_ANSWER_NUMBER_DEVIATION"]

    BLEU_COLUMNS = ["ID", "GENERATED", "GROUND_TRUTH", "BLEU_SCORE"]

    PREDICTIONS_FILENAME = "predictions.pkl"
    LOG_FILENAME = "log.txt"

    STATISTICS_FILENAME = "statistics.csv"
    QUESTION_BLEU_FILENAME = "question_bleu_scores.csv"
    ANSWER_BLEU_FILENAME = "answer_bleu_scores.csv"

    N_GRAMS_WEIGHTS = (0.25, 0.25, 0.25, 0.25)


    # ------------
    # Constructor
    # ------------
    def __init__(self):
        self.experiment_id = None
        self.predictions = pd.DataFrame(columns=self.PREDICTION_COLUMNS)
        self.log = pd.DataFrame()

        self.questionnaire_id = None

        self.conversion_error = False
        self.generated_questionnaires = 0
        self.questions_with_missing_answers = 0

        self.generated_question_number = 0
        self.question_number_deviation = 0
        
        self.avg_generated_answer_number = 0
        self.avg_answer_number_deviation = 0
        
        self.statistics = pd.DataFrame(columns=self.STATISTICS_COLUMNS)
        
        self.question_bleu_scores = pd.DataFrame(columns=self.BLEU_COLUMNS)
        self.answer_bleu_scores = pd.DataFrame(columns=self.BLEU_COLUMNS)

    
    # ------------
    # Methos
    # ------------
    def load_data(self, project_root, experiment_id):
        result_dir_path = os.path.join(project_root, "models", experiment_id)

        predictions_path = os.path.join(result_dir_path, self.PREDICTIONS_FILENAME)
        self.predictions = pd.read_pickle(predictions_path)

        log_path = os.path.join(result_dir_path, self.LOG_FILENAME)
        with open(log_path, 'r') as file:
            log_data = file.read()
        self.log = log_data


    def set_questionnaire_id(self, questionnaire_id):
        self.questionnaire_id = questionnaire_id


    def clear(self):
        self.questionnaire_id = 0

        self.conversion_error = False
        self.generated_questionnaires = 0
        self.questions_with_missing_answers = 0
        
        self.generated_question_number = 0
        self.question_number_deviation = 0
        
        self.avg_generated_answer_number = 0
        self.avg_answer_number_deviation = 0
        
        self.question_bleu_scores = pd.DataFrame(columns=self.BLEU_COLUMNS)
        self.answer_bleu_scores = pd.DataFrame(columns=self.BLEU_COLUMNS)


    def compute_statistics(self, results_dir):
        for i, _ in enumerate(self.predictions):
            id = self.predictions["QUESTIONNAIRE_ID"].values[i]
            self.set_questionnaire_id(id)
            self._compute_deviations()
            
            self.statistics = pd.concat([self.statistics, pd.DataFrame({
                "QUESTIONNAIRE_ID": [id],
                "GENERATED_QUESTION_NUMBER": [self.generated_question_number],
                "QUESTION_NUMBER_DEVIATION": [self.question_number_deviation],
                "AVERAGE_GENERATED_ANSWER_NUMBER": [self.avg_generated_answer_number],
                "AVERAGE_ANSWER_NUMBER_DEVIATION": [self.avg_answer_number_deviation]
            })], ignore_index=True)

            self.clear()

        self.statistics.to_csv(os.path.join(results_dir, self.STATISTICS_FILENAME), index=False)
    

    def _check_json_integrity(self):
        self.conversion_error, self.generated_questionnaires, self.questions_with_missing_answers = TFQuestionnairesDataset.check_json_integrity(self.questionnaire_id, self.predictions)


    def _compute_deviations(self):
        pred = self.predictions[self.predictions["QUESTIONNAIRE_ID"] == self.questionnaire_id]

        ground_truth = TFQuestionnairesDataset.from_json(questionnaire_id=self.questionnaire_id, json_data=pred["GROUND_TRUTH_JSON"].values[0])
        predicted = TFQuestionnairesDataset.from_json(questionnaire_id=self.questionnaire_id, json_data=pred["PREDICTED_JSON"].values[0])

        # Question number deviation
        self.generated_question_number = len(predicted.questions)
        ground_truth_question_number = len(ground_truth.questions)
        self.question_number_deviation = ground_truth_question_number - self.generated_question_number

        # Answer number deviation
        self.avg_generated_answer_number = predicted.get_average_answer_number(self.questionnaire_id)
        avg_ground_truth_answer_number = ground_truth.get_average_answer_number(self.questionnaire_id)
        self.avg_answer_number_deviation = avg_ground_truth_answer_number - self.avg_generated_answer_number


    def print_current_deviation_results(self):
        print("===========================================")
        print(f"    QUESTIONNAIRE {self.questionnaire_id}")
        print("===========================================")
        print("Generated Question Number: ", self.generated_question_number)
        print("Question Number Deviation: ", self.question_number_deviation)
        print("Average Generated Answer Number: ", self.avg_generated_answer_number)
        print("Average Answer Number Deviation: ", self.avg_answer_number_deviation)
        print("===========================================")

    
    def compute_bleu_scores(self, results_dir):
        for i, _ in enumerate(self.predictions):
            id = self.predictions["QUESTIONNAIRE_ID"].values[i]
            self.set_questionnaire_id(id)
            self._compute_bleu_scores()

        self.question_bleu_scores.to_csv(os.path.join(results_dir, self.QUESTION_BLEU_FILENAME), index=False)
        self.answer_bleu_scores.to_csv(os.path.join(results_dir, self.ANSWER_BLEU_FILENAME), index=False)


    def _compute_bleu_scores(self):
        pred = self.predictions[self.predictions["QUESTIONNAIRE_ID"] == self.questionnaire_id]

        ground_truth = TFQuestionnairesDataset.from_json(questionnaire_id=self.questionnaire_id, json_data=pred["GROUND_TRUTH_JSON"].values[0])
        generated = TFQuestionnairesDataset.from_json(questionnaire_id=self.questionnaire_id, json_data=pred["PREDICTED_JSON"].values[0])

        for question in generated.questions.iterrows():
            qst_id = question[1]["ID"]
            qst_text = question[1]["NAME"]

            ground_truth_answers = ground_truth.answers[ground_truth.answers["QUESTION_ID"] == qst_id]

            self.question_bleu_scores = pd.concat([self.question_bleu_scores, self._compute_candidate_blue_score(qst_id, qst_text, ground_truth.questions["NAME"])], ignore_index=True)

            for answer in generated.answers[generated.answers["QUESTION_ID"] == qst_id].iterrows():
                ans_id = answer[1]["ID"]
                ans_text = answer[1]["ANSWER"]
                self.answer_bleu_scores = pd.concat([self.answer_bleu_scores, self._compute_candidate_blue_score(ans_id, ans_text, ground_truth_answers["ANSWER"])], ignore_index=True)
    

    def _compute_candidate_blue_score(self, id, candidate, ground_truth_references):
        question_split = candidate.split()

        df = pd.DataFrame(columns=self.BLEU_COLUMNS)

        for reference in ground_truth_references:
            reference_split = [reference.split()]
            bleu_core = sentence_bleu(reference_split, question_split, smoothing_function=SmoothingFunction().method4, weights=self.N_GRAMS_WEIGHTS)

            new_row = pd.DataFrame({
                "ID": [id],
                "GENERATED": [candidate],
                "GROUND_TRUTH": [reference],
                "BLEU_SCORE": [bleu_core]
            })

            df = pd.concat([df, new_row], ignore_index=True)

        return df
