import pandas as pd
import os
from src.data.TFQuestionnairesDataset import TFQuestionnairesDataset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

class ModelEvaluator:
    # ------------
    # Constants
    # ------------
    PREDICTION_COLUMNS = ["QUESTIONNAIRE_ID", "GROUND_TRUTH_JSON", "PREDICTED_JSON", "REPORTED_EXCEPTION", 
                      "RESPONSE_TIME", "PROMPT_TOKENS", "COMPLETITION_TOKENS", "TOTAL_TOKENS",
                      "CONVERSION_ERROR"]
    
    STATISTICS_COLUMNS = ["QUESTIONNAIRE_ID", 
                          "CONVERSION_ERROR", "IS_JSON", "ERROR_MESSAGE", "GENERATED_QUESTIONNAIRES", "QUESTIONS_WITH_MISSING_ANSWERS", 
                          "GENERATED_QUESTION_NUMBER", "QUESTION_NUMBER_DEVIATION", 
                          "AVERAGE_GENERATED_ANSWER_NUMBER", "AVERAGE_ANSWER_NUMBER_DEVIATION"]

    BLEU_COLUMNS_QUESTION = ["ID", "GENERATED", "GROUND_TRUTH", "BLEU_SCORE"]
    BLEU_COLUMNS_ANSWER = ["ID", "QUESTION_ID", "GENERATED", "GROUND_TRUTH", "BLEU_SCORE"]

    ROUGE_COLUMNS_QUESTION = ["ID", "GENERATED", "GROUND_TRUTH",
                              "R1_PRECISION", "R1_RECALL", "R1_F1_SCORE",
                              "R2_PRECISION", "R2_RECALL", "R2_F1_SCORE",
                              "RL_PRECISION", "RL_RECALL", "RL_F1_SCORE"]
    ROUGE_COLUMNS_ANSWER = ["ID", "QUESTION_ID", "GENERATED", "GROUND_TRUTH",
                            "R1_PRECISION", "R1_RECALL", "R1_F1_SCORE",
                            "R2_PRECISION", "R2_RECALL", "R2_F1_SCORE",
                            "RL_PRECISION", "RL_RECALL", "RL_F1_SCORE"]

    PREDICTIONS_FILENAME = "predictions.pkl"
    LOG_FILENAME = "log.txt"

    STATISTICS_FILENAME = "statistics.csv"
    QUESTION_BLEU_FILENAME = "question_bleu_scores"
    ANSWER_BLEU_FILENAME = "answer_bleu_scores"

    QUESTION_ROUGE_FILENAME = "question_rouge_scores"
    ANSWER_ROUGE_FILENAME = "answer_rouge_scores"

    N_GRAMS_WEIGHTS = (0.5, 0.5, 0, 0)



    # ------------
    # Constructor
    # ------------
    def __init__(self):
        self.experiment_id = None
        self.predictions = pd.DataFrame(columns=self.PREDICTION_COLUMNS)
        self.log = pd.DataFrame()

        self.questionnaire_id = None

        self.conversion_error = False
        self.is_json = True
        self.error_message = ""
        self.generated_questionnaires = -1
        self.questions_with_missing_answers = -1

        self.generated_question_number = -1
        self.question_number_deviation = -1
        
        self.avg_generated_answer_number = -1
        self.avg_answer_number_deviation = -1
        
        self.statistics = pd.DataFrame(columns=self.STATISTICS_COLUMNS)
        
        self.question_bleu_scores = pd.DataFrame(columns=self.BLEU_COLUMNS_QUESTION)
        self.answer_bleu_scores = pd.DataFrame(columns=self.BLEU_COLUMNS_ANSWER)

        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.question_rouge_scores = pd.DataFrame(columns=self.ROUGE_COLUMNS_QUESTION)
        self.answer_rouge_scores = pd.DataFrame(columns=self.ROUGE_COLUMNS_ANSWER)

    
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
        self.is_json = True
        self.error_message = ""
        self.generated_questionnaires = -1
        self.questions_with_missing_answers = -1
        
        self.generated_question_number = -1
        self.question_number_deviation = -1
        
        self.avg_generated_answer_number = -1
        self.avg_answer_number_deviation = -1
        
        self.question_bleu_scores = pd.DataFrame(columns=self.BLEU_COLUMNS_QUESTION)
        self.answer_bleu_scores = pd.DataFrame(columns=self.BLEU_COLUMNS_ANSWER)

        self.question_rouge_scores = pd.DataFrame(columns=self.ROUGE_COLUMNS_QUESTION)
        self.answer_rouge_scores = pd.DataFrame(columns=self.ROUGE_COLUMNS_ANSWER)


    def compute_statistics(self, project_root, results_dir):
        for pred in self.predictions.iterrows():
            id = pred[1]["QUESTIONNAIRE_ID"]
            self.set_questionnaire_id(id)

            self._check_json_integrity(project_root, pred[1])
            if self.is_json:
                self._compute_deviations(project_root, pred[1])

            self.statistics = pd.concat([self.statistics, pd.DataFrame({
                "QUESTIONNAIRE_ID": [id],
                "CONVERSION_ERROR": [self.conversion_error],
                "IS_JSON": [self.is_json],
                "ERROR_MESSAGE": [self.error_message],
                "GENERATED_QUESTIONNAIRES": [self.generated_questionnaires],
                "QUESTIONS_WITH_MISSING_ANSWERS": [self.questions_with_missing_answers],
                "GENERATED_QUESTION_NUMBER": [self.generated_question_number],
                "QUESTION_NUMBER_DEVIATION": [self.question_number_deviation],
                "AVERAGE_GENERATED_ANSWER_NUMBER": [self.avg_generated_answer_number],
                "AVERAGE_ANSWER_NUMBER_DEVIATION": [self.avg_answer_number_deviation]
            })], ignore_index=True)

            self.clear()

        self.statistics.to_csv(os.path.join(results_dir, self.STATISTICS_FILENAME), index=False)
    

    def _check_json_integrity(self, project_root, pred):
        result = TFQuestionnairesDataset.check_json_integrity(project_root, pred["PREDICTED_JSON"])

        self.conversion_error = result["conversion_error"]
        self.is_json = result["is_json"]
        self.error_message = result["error_message"]
        self.generated_questionnaires = result["generated_questionnaires"]
        self.questions_with_missing_answers = result["questions_with_missing_answers"]


    def _compute_deviations(self, project_root, pred):
        ground_truth = TFQuestionnairesDataset.from_json(project_root=project_root, questionnaire_id=self.questionnaire_id, json_data=pred["GROUND_TRUTH_JSON"])
        predicted = TFQuestionnairesDataset.from_json(project_root=project_root, questionnaire_id=self.questionnaire_id, json_data=pred["PREDICTED_JSON"])

        # Question number deviation
        self.generated_question_number = len(predicted.questions)
        ground_truth_question_number = len(ground_truth.questions)
        self.question_number_deviation = ground_truth_question_number - self.generated_question_number

        # Answer number deviation
        self.avg_generated_answer_number = predicted.get_average_answer_number(self.questionnaire_id)
        avg_ground_truth_answer_number = ground_truth.get_average_answer_number(self.questionnaire_id)
        self.avg_answer_number_deviation = avg_ground_truth_answer_number - self.avg_generated_answer_number


    def compute_bleu_scores(self, project_root, results_dir):
        bleu_scores_path = os.path.join(results_dir, "BLEU_Scores")
            
        if not os.path.exists(bleu_scores_path):
            os.makedirs(bleu_scores_path)

        for pred in self.predictions.iterrows():
            id = pred[1]["QUESTIONNAIRE_ID"]
            self.set_questionnaire_id(id)
            self._compute_bleu_scores(project_root, pred[1])

            if not self.question_bleu_scores.empty:
                question_filename = f"{self.QUESTION_BLEU_FILENAME}__QST_{id}.csv"
                self.question_bleu_scores.to_csv(os.path.join(bleu_scores_path, question_filename), index=False)

            if not self.answer_bleu_scores.empty:
                answer_filename = f"{self.ANSWER_BLEU_FILENAME}__QST_{id}.csv"
                self.answer_bleu_scores.to_csv(os.path.join(bleu_scores_path, answer_filename), index=False)
            
            self.clear()


    def _compute_bleu_scores(self, project_root, pred):
        try:
            ground_truth = TFQuestionnairesDataset.from_json(project_root=project_root, questionnaire_id=self.questionnaire_id, json_data=pred["GROUND_TRUTH_JSON"])
            generated = TFQuestionnairesDataset.from_json(project_root=project_root, questionnaire_id=self.questionnaire_id, json_data=pred["PREDICTED_JSON"])

            for question in generated.questions.iterrows():
                qst_id = question[1]["ID"]
                qst_text = question[1]["NAME"]
                ground_truth_answers = ground_truth.answers[ground_truth.answers["QUESTION_ID"] == qst_id]
                self.question_bleu_scores = pd.concat([self.question_bleu_scores, self._compute_candidate_blue_score(qst_id, qst_text, ground_truth.questions["NAME"])], ignore_index=True)

                for answer in generated.answers[generated.answers["QUESTION_ID"] == qst_id].iterrows():
                    ans_id = answer[1]["ID"]
                    ans_text = answer[1]["ANSWER"]
                    self.answer_bleu_scores = pd.concat([self.answer_bleu_scores, 
                                                         self._compute_candidate_blue_score(ans_id, ans_text, ground_truth_answers["ANSWER"], is_question=False, question_id=qst_id)
                                                        ], ignore_index=True)
        
        except Exception as e:
            return


    def _compute_candidate_blue_score(self, id, candidate, ground_truth_references, is_question=True, question_id=None):
        question_split = candidate.split()

        df = pd.DataFrame(columns=self.BLEU_COLUMNS_QUESTION)

        for reference in ground_truth_references:
            reference_split = [reference.split()]
            bleu_core = sentence_bleu(reference_split, question_split, smoothing_function=SmoothingFunction().method4, weights=self.N_GRAMS_WEIGHTS)

            if is_question:
                new_row = pd.DataFrame({
                    "ID": [id],
                    "GENERATED": [candidate],
                    "GROUND_TRUTH": [reference],
                    "BLEU_SCORE": [bleu_core]
                })
            else:
                new_row = pd.DataFrame({
                    "ID": [id],
                    "QUESTION_ID": question_id,
                    "GENERATED": [candidate],
                    "GROUND_TRUTH": [reference],
                    "BLEU_SCORE": [bleu_core]
                })

            df = pd.concat([df, new_row], ignore_index=True)

        return df


    def compute_rouge_scores(self, project_root, results_dir):
        rouge_scores_path = os.path.join(results_dir, "ROUGE_Scores")
            
        if not os.path.exists(rouge_scores_path):
            os.makedirs(rouge_scores_path)

        for pred in self.predictions.iterrows():
            id = pred[1]["QUESTIONNAIRE_ID"]
            self.set_questionnaire_id(id)
            self._compute_rouge_scores(project_root, pred[1])

            if not self.question_rouge_scores.empty:
                question_filename = f"{self.QUESTION_ROUGE_FILENAME}__QST_{id}.csv"
                self.question_rouge_scores.to_csv(os.path.join(rouge_scores_path, question_filename), index=False)

            if not self.answer_rouge_scores.empty:
                answer_filename = f"{self.ANSWER_ROUGE_FILENAME}__QST_{id}.csv"
                self.answer_rouge_scores.to_csv(os.path.join(rouge_scores_path, answer_filename), index=False)
            
            self.clear()


    def _compute_rouge_scores(self, project_root, pred):
        try:
            ground_truth = TFQuestionnairesDataset.from_json(project_root=project_root, questionnaire_id=self.questionnaire_id, json_data=pred["GROUND_TRUTH_JSON"])
            generated = TFQuestionnairesDataset.from_json(project_root=project_root, questionnaire_id=self.questionnaire_id, json_data=pred["PREDICTED_JSON"])

            for question in generated.questions.iterrows():
                qst_id = question[1]["ID"]
                qst_text = question[1]["NAME"]
                ground_truth_answers = ground_truth.answers[ground_truth.answers["QUESTION_ID"] == qst_id]
                self.question_rouge_scores = pd.concat([self.question_rouge_scores, 
                                                        self._compute_candidate_rouge_score(qst_id, qst_text, ground_truth.questions["NAME"])
                                                       ], ignore_index=True)

                for answer in generated.answers[generated.answers["QUESTION_ID"] == qst_id].iterrows():
                    ans_id = answer[1]["ID"]
                    ans_text = answer[1]["ANSWER"]
                    self.answer_rouge_scores = pd.concat([self.answer_rouge_scores, 
                                                          self._compute_candidate_rouge_score(ans_id, ans_text, ground_truth_answers["ANSWER"], is_question=False, question_id=qst_id)
                                                         ], ignore_index=True)
        
        except Exception as e:
            return


    def _compute_candidate_rouge_score(self, id, candidate, ground_truth_references, is_question=True, question_id=None):
        df = pd.DataFrame(columns=self.ROUGE_COLUMNS_QUESTION)

        for reference in ground_truth_references:
            scores = self.scorer.score(reference, candidate)
            
            if is_question:
                new_row = pd.DataFrame({
                    "ID": [id],
                    "GENERATED": [candidate],
                    "GROUND_TRUTH": [reference],
                    "R1_PRECISION": [scores["rouge1"].precision],
                    "R1_RECALL": [scores["rouge1"].recall],
                    "R1_F1_SCORE": [scores["rouge1"].fmeasure],
                    "R2_PRECISION": [scores["rouge2"].precision],
                    "R2_RECALL": [scores["rouge2"].recall],
                    "R2_F1_SCORE": [scores["rouge2"].fmeasure],
                    "RL_PRECISION": [scores["rougeL"].precision],
                    "RL_RECALL": [scores["rougeL"].recall],
                    "RL_F1_SCORE": [scores["rougeL"].fmeasure]
                })
            else:
                new_row = pd.DataFrame({
                    "ID": [id],
                    "QUESTION_ID": question_id,
                    "GENERATED": [candidate],
                    "GROUND_TRUTH": [reference],
                    "R1_PRECISION": [scores["rouge1"].precision],
                    "R1_RECALL": [scores["rouge1"].recall],
                    "R1_F1_SCORE": [scores["rouge1"].fmeasure],
                    "R2_PRECISION": [scores["rouge2"].precision],
                    "R2_RECALL": [scores["rouge2"].recall],
                    "R2_F1_SCORE": [scores["rouge2"].fmeasure],
                    "RL_PRECISION": [scores["rougeL"].precision],
                    "RL_RECALL": [scores["rougeL"].recall],
                    "RL_F1_SCORE": [scores["rougeL"].fmeasure]
                })

            df = pd.concat([df, new_row], ignore_index=True)

        return df