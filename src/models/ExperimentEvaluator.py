import pandas as pd
import os
from src.data.TFQuestionnairesDataset import TFQuestionnairesDataset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import numpy as np


class ExperimentEvaluator:
    # ------------
    # Constants
    # ------------
    PREDICTION_COLUMNS = ["QUESTIONNAIRE_ID", "SAMPLE_QUESTIONNAIRES_IDS", 
                          "GROUND_TRUTH_JSON", "PREDICTED_JSON", "REPORTED_EXCEPTION",
                          "RESPONSE_TIME", "PROMPT_TOKENS", "COMPLETITION_TOKENS", "TOTAL_TOKENS",
                          "CONVERSION_ERROR"]
    
    STATISTICS_COLUMNS = ["QUESTIONNAIRE_ID", 
                          "CONVERSION_ERROR", "IS_JSON", "ERROR_MESSAGE", 
                          "GENERATED_QUESTIONNAIRES", 
                          "QUESTIONS_WITH_MISSING_ANSWERS", "NOT_UNIOUE_QUESTION_CODES", 
                          "GENERATED_QUESTION_NUMBER", "QUESTION_NUMBER_DEVIATION", "QUESTION_NUMBER_DEVIATION_FROM_SAMPLE",
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

    BLEU_SCORE_N_GRAMS_WEIGHTS = (0.5, 0.5, 0, 0)
    SYNT_SIM_BLEU_SCORE_N_GRAMS_WEIGHTS = (0, 0, 0, 1)

    SYNTACTIC_SIMILARITY_INTRAQSTN_COLUMNS = ["QUESTIONNAIRE_ID", "BLEU_SCORE", "ROUGE_L_PRECISION", "ROUGE_L_RECALL", "ROUGE_L_F1"]
    SYNTACTIC_SIMILARITY_SAMPLE_COLUMNS = ["QUESTIONNAIRE_ID", "SAMPLE_QUESTIONNAIRE_ID", "BLEU_SCORE", "ROUGE_L_PRECISION", "ROUGE_L_RECALL", "ROUGE_L_F1"]
    

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

        self.generated_question_number = -1
        self.question_number_deviation = -1
        self.question_number_deviation_from_sample = -1
        self.questions_with_missing_answers = -1
        self.not_unique_question_codes = -1
        
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
        self.experiment_id = experiment_id
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

        self.generated_question_number = -1
        self.question_number_deviation = -1
        self.question_number_deviation_from_sample = -1
        self.questions_with_missing_answers = -1
        self.not_unique_question_codes = -1

        self.avg_generated_answer_number = -1
        self.avg_answer_number_deviation = -1
        
        self.question_bleu_scores = pd.DataFrame(columns=self.BLEU_COLUMNS_QUESTION)
        self.answer_bleu_scores = pd.DataFrame(columns=self.BLEU_COLUMNS_ANSWER)

        self.question_rouge_scores = pd.DataFrame(columns=self.ROUGE_COLUMNS_QUESTION)
        self.answer_rouge_scores = pd.DataFrame(columns=self.ROUGE_COLUMNS_ANSWER)


    def compute_statistics(self, project_root, results_dir):
        dataset = TFQuestionnairesDataset()
        dataset.load_data(project_root=project_root)

        for pred in self.predictions.iterrows():
            id = pred[1]["QUESTIONNAIRE_ID"]
            self.set_questionnaire_id(id)

            self._check_json_integrity(project_root, pred[1])
            if self.is_json:
                self._compute_statistics(project_root, pred[1], dataset)

            self.statistics = pd.concat([self.statistics, pd.DataFrame({
                "QUESTIONNAIRE_ID": [id],
                "CONVERSION_ERROR": [self.conversion_error],
                "IS_JSON": [self.is_json],
                "ERROR_MESSAGE": [self.error_message],
                "GENERATED_QUESTIONNAIRES": [self.generated_questionnaires],
                "QUESTIONS_WITH_MISSING_ANSWERS": [self.questions_with_missing_answers],
                "GENERATED_QUESTION_NUMBER": [self.generated_question_number],
                "NOT_UNIOUE_QUESTION_CODES": [self.not_unique_question_codes],
                "QUESTION_NUMBER_DEVIATION": [self.question_number_deviation],
                "QUESTION_NUMBER_DEVIATION_FROM_SAMPLE": [self.question_number_deviation_from_sample],
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


    def _compute_statistics(self, project_root, pred, dataset):
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

        # Not unique question codes
        self.not_unique_question_codes = predicted.get_not_unique_question_codes()

        # Question number deviation from sample (only few-shot)
        if self.experiment_id.startswith("1s") and "SAMPLE_QUESTIONNAIRES_IDS" in pred.index:
            sample_ids = pred["SAMPLE_QUESTIONNAIRES_IDS"]
            deviations = []
            
            for sample_id in sample_ids:
                sample_questionnaire = dataset.get_questionnaire_data(sample_id)
                sample_question_number = len(sample_questionnaire.questions)
                deviations.append(sample_question_number - self.generated_question_number)

            self.question_number_deviation_from_sample = np.mean(deviations)
       

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
            bleu_core = sentence_bleu(reference_split, question_split, smoothing_function=SmoothingFunction().method4, weights=self.BLEU_SCORE_N_GRAMS_WEIGHTS)

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
    

    def compute_bleu_score_distribution(self, results_dir):
        bleu_scores_path = os.path.join(results_dir, "BLEU_Scores")

        questionnaires_scores = pd.DataFrame(columns=["QUESTIONNAIRE_ID"])
        max_questions_scores = pd.DataFrame(columns=["QUESTIONNAIRE_ID", "ID"])
        max_answers_scores = pd.DataFrame(columns=["QUESTIONNAIRE_ID", "QUESTION_ID", "ID"])

        for qst_id in self.predictions["QUESTIONNAIRE_ID"]:
            questionnaire_new_row = pd.DataFrame(columns=["QUESTIONNAIRE_ID"])
            questionnaire_new_row["QUESTIONNAIRE_ID"] = [qst_id]
            
            for scores_csv in os.listdir(bleu_scores_path):
                if str(qst_id) in scores_csv:
                    if "question" in scores_csv:
                        global_questions_scores = pd.read_csv(os.path.join(bleu_scores_path, scores_csv))

                        max_questions_scores_df = ExperimentEvaluator._get_max_bleu_score(global_questions_scores)
                        qst_distribution = ExperimentEvaluator._get_distribution_bleu_score(global_questions_scores)
                        max_questions_scores_df = ExperimentEvaluator._concat_bleu_scores_and_distribution(max_questions_scores_df, qst_distribution)

                        distribution = ExperimentEvaluator._get_distribution_bleu_score(max_questions_scores_df, at_questionnaire_level=True)
                        questionnaire_new_row = ExperimentEvaluator._concat_bleu_scores_and_distribution(questionnaire_new_row, distribution, at_questionnaire_level=True)

                        max_questions_scores_df["QUESTIONNAIRE_ID"] = [qst_id] * len(max_questions_scores_df)
                        max_questions_scores = pd.concat([max_questions_scores, max_questions_scores_df], ignore_index=True)

                    elif "answer" in scores_csv:
                        global_answers_scores = pd.read_csv(os.path.join(bleu_scores_path, scores_csv))

                        max_answers_scores_df = ExperimentEvaluator._get_max_bleu_score(global_answers_scores, is_question=False)
                        asw_distribution = ExperimentEvaluator._get_distribution_bleu_score(global_answers_scores, is_question=False)
                        max_answers_scores_df = ExperimentEvaluator._concat_bleu_scores_and_distribution(max_answers_scores_df, asw_distribution, is_question=False)

                        distribution =  ExperimentEvaluator._get_distribution_bleu_score(max_answers_scores_df, at_questionnaire_level=True)
                        questionnaire_new_row = ExperimentEvaluator._concat_bleu_scores_and_distribution(questionnaire_new_row, distribution, at_questionnaire_level=True, is_question=False)

                        max_answers_scores_df["QUESTIONNAIRE_ID"] = [qst_id] * len(max_answers_scores_df)
                        max_answers_scores = pd.concat([max_answers_scores, max_answers_scores_df], ignore_index=True)

            questionnaires_scores = pd.concat([questionnaires_scores, questionnaire_new_row], ignore_index=True)

        questionnaires_scores.to_csv(os.path.join(results_dir, "BLEU_questionnaires.csv"), index=False)
        max_questions_scores.to_csv(os.path.join(results_dir, "BLEU_questions.csv"), index=False)
        max_answers_scores.to_csv(os.path.join(results_dir, "BLEU_answers.csv"), index=False)


    def _get_max_bleu_score(scores_df, is_question=True):
        if is_question:
            max_ids = scores_df.groupby('ID')['BLEU_SCORE'].idxmax()
        else:
            max_ids = scores_df.groupby(['QUESTION_ID', 'ID'])['BLEU_SCORE'].idxmax()
        return scores_df.loc[max_ids]
    

    def _get_distribution_bleu_score(scores_df, at_questionnaire_level=False, is_question=True):
        if at_questionnaire_level:
            mean = scores_df['BLEU_SCORE'].mean()
            variance = scores_df['BLEU_SCORE'].var()
        else:
            if is_question:
                mean = scores_df.groupby('ID')['BLEU_SCORE'].mean()
                variance = scores_df.groupby('ID')['BLEU_SCORE'].var()
            else:
                mean = scores_df.groupby(['QUESTION_ID', 'ID'])['BLEU_SCORE'].mean()
                variance = scores_df.groupby(['QUESTION_ID', 'ID'])['BLEU_SCORE'].var()
        return {
            "mean": mean, 
            "variance": variance
            }
    

    def _concat_bleu_scores_and_distribution(scores_df, distribution, at_questionnaire_level=False, is_question=True):
        if at_questionnaire_level:
            if is_question:
                scores_df["QST_BLEU_SCORE_MEAN"] = distribution['mean']
                scores_df["QST_BLEU_SCORE_VAR"] = distribution['variance']
            else:
                scores_df["ASW_BLEU_SCORE_MEAN"] = distribution['mean']
                scores_df["ASW_BLEU_SCORE_VAR"] = distribution['variance']

        else:
            if is_question:
                scores_df = scores_df.merge(distribution['mean'], on='ID', suffixes=('', '_MEAN'))
                scores_df = scores_df.merge(distribution['variance'], on='ID', suffixes=('', '_VAR'))
            else:
                scores_df = scores_df.merge(distribution['mean'], on=['QUESTION_ID', 'ID'], suffixes=('', '_MEAN'))
                scores_df = scores_df.merge(distribution['variance'], on=['QUESTION_ID', 'ID'], suffixes=('', '_VAR'))

        return scores_df
        
    
    def _get_max_rouge_score(scores_df, is_question=True):
        if is_question:
            max_ids = scores_df.groupby('ID')['RL_F1_SCORE'].idxmax()
        else:
            max_ids = scores_df.groupby(['QUESTION_ID', 'ID'])['RL_F1_SCORE'].idxmax()
        return scores_df.loc[max_ids]
    

    def _get_distribution_rouge_score(scores_df, at_questionnaire_level=False, is_question=True):
        if at_questionnaire_level:
            mean = scores_df['RL_F1_SCORE'].mean()
            variance = scores_df['RL_F1_SCORE'].var()
        else:
            if is_question:
                mean = scores_df.groupby('ID')['RL_F1_SCORE'].mean()
                variance = scores_df.groupby('ID')['RL_F1_SCORE'].var()
            else:
                mean = scores_df.groupby(['QUESTION_ID', 'ID'])['RL_F1_SCORE'].mean()
                variance = scores_df.groupby(['QUESTION_ID', 'ID'])['RL_F1_SCORE'].var()
        return {
            "mean": mean, 
            "variance": variance
            }
    

    def _concat_rouge_scores_and_distribution(scores_df, distribution, at_questionnaire_level=False, is_question=True):
        if at_questionnaire_level:
            if is_question:
                scores_df["QST_ROUGE_SCORE_MEAN"] = distribution['mean']
                scores_df["QST_ROUGE_SCORE_VAR"] = distribution['variance']
            else:
                scores_df["ASW_ROUGE_SCORE_MEAN"] = distribution['mean']
                scores_df["ASW_ROUGE_SCORE_VAR"] = distribution['variance']

        else:
            if is_question:
                scores_df = scores_df.merge(distribution['mean'], on='ID', suffixes=('', '_MEAN'))
                scores_df = scores_df.merge(distribution['variance'], on='ID', suffixes=('', '_VAR'))
            else:
                scores_df = scores_df.merge(distribution['mean'], on=['QUESTION_ID', 'ID'], suffixes=('', '_MEAN'))
                scores_df = scores_df.merge(distribution['variance'], on=['QUESTION_ID', 'ID'], suffixes=('', '_VAR'))

        return scores_df
        

    def compute_rouge_score_distribution(self, results_dir):
        rouge_scores_path = os.path.join(results_dir, "ROUGE_Scores")

        columns_to_discard = ["R1_PRECISION", "R1_RECALL", "R1_F1_SCORE", "R2_PRECISION", "R2_RECALL", "R2_F1_SCORE", "RL_PRECISION", "RL_RECALL"]

        questionnaires_scores = pd.DataFrame(columns=["QUESTIONNAIRE_ID"])
        max_questions_scores = pd.DataFrame(columns=["QUESTIONNAIRE_ID", "ID"])
        max_answers_scores = pd.DataFrame(columns=["QUESTIONNAIRE_ID", "QUESTION_ID", "ID"])

        for qst_id in self.predictions["QUESTIONNAIRE_ID"]:
            questionnaire_new_row = pd.DataFrame(columns=["QUESTIONNAIRE_ID"])
            questionnaire_new_row["QUESTIONNAIRE_ID"] = [qst_id]
            
            for scores_csv in os.listdir(rouge_scores_path):
                if str(qst_id) in scores_csv:
                    if "question" in scores_csv:
                        global_questions_scores = pd.read_csv(os.path.join(rouge_scores_path, scores_csv))
                        global_questions_scores = global_questions_scores.drop(columns=columns_to_discard, axis=1)

                        max_questions_scores_df = ExperimentEvaluator._get_max_rouge_score(global_questions_scores)
                        qst_distribution = ExperimentEvaluator._get_distribution_rouge_score(global_questions_scores)
                        max_questions_scores_df = ExperimentEvaluator._concat_rouge_scores_and_distribution(max_questions_scores_df, qst_distribution)

                        distribution = ExperimentEvaluator._get_distribution_rouge_score(max_questions_scores_df, at_questionnaire_level=True)
                        questionnaire_new_row = ExperimentEvaluator._concat_rouge_scores_and_distribution(questionnaire_new_row, distribution, at_questionnaire_level=True)

                        max_questions_scores_df["QUESTIONNAIRE_ID"] = [qst_id] * len(max_questions_scores_df)
                        max_questions_scores = pd.concat([max_questions_scores, max_questions_scores_df], ignore_index=True)

                    elif "answer" in scores_csv:
                        global_answers_scores = pd.read_csv(os.path.join(rouge_scores_path, scores_csv))
                        global_answers_scores = global_answers_scores.drop(columns=columns_to_discard, axis=1)

                        max_answers_scores_df = ExperimentEvaluator._get_max_rouge_score(global_answers_scores, is_question=False)
                        asw_distribution = ExperimentEvaluator._get_distribution_rouge_score(global_answers_scores, is_question=False)
                        max_answers_scores_df = ExperimentEvaluator._concat_rouge_scores_and_distribution(max_answers_scores_df, asw_distribution, is_question=False)

                        distribution = ExperimentEvaluator._get_distribution_rouge_score(max_answers_scores_df, at_questionnaire_level=True)
                        questionnaire_new_row = ExperimentEvaluator._concat_rouge_scores_and_distribution(questionnaire_new_row, distribution, at_questionnaire_level=True, is_question=False)

                        max_answers_scores_df["QUESTIONNAIRE_ID"] = [qst_id] * len(max_answers_scores_df)
                        max_answers_scores = pd.concat([max_answers_scores, max_answers_scores_df], ignore_index=True)

            questionnaires_scores = pd.concat([questionnaires_scores, questionnaire_new_row], ignore_index=True)

        questionnaires_scores.to_csv(os.path.join(results_dir, "ROUGE_L_F1_questionnaires.csv"), index=False)
        max_questions_scores.to_csv(os.path.join(results_dir, "ROUGE_L_F1_questions.csv"), index=False)
        max_answers_scores.to_csv(os.path.join(results_dir, "ROUGE_L_F1_answers.csv"), index=False)


    def compute_syntactic_similarities(self, project_root, results_dir):
        # Intraquestionnaire Syntactic Similarity
        intraquestionnaire_sintactic_similarity = self._compute_intraquestionnaires_syntactic_similarity(project_root)
        intraquestionnaire_sintactic_similarity.to_csv(os.path.join(results_dir, "Intraquestionnaire_Syntactic_Similarity.csv"), index=False)

        # Syntactic Similarity with Sample Questionnaires (only for few-shot)
        if self.experiment_id.startswith("1s") and "SAMPLE_QUESTIONNAIRES_IDS" in self.predictions.columns:
            dataset = TFQuestionnairesDataset()
            dataset.load_data(project_root=project_root)

            syntactic_similarity_with_sample = self._compute_syntactic_similarity_with_samples(project_root=project_root, dataset=dataset)
            syntactic_similarity_with_sample.to_csv(os.path.join(results_dir, "Syntactic_Similarity_With_Sample.csv"), index=False)


    def _compute_intraquestionnaires_syntactic_similarity(self, project_root):
        sintactic_similarity_df = pd.DataFrame(columns=self.SYNTACTIC_SIMILARITY_INTRAQSTN_COLUMNS)
        
        for pred in self.predictions.iterrows():
            id = pred[1]["QUESTIONNAIRE_ID"]
            self.set_questionnaire_id(id)
            new_row = self._compute_intraquestionnaire_syntactic_similarity(project_root, pred[1])
            sintactic_similarity_df = pd.concat([sintactic_similarity_df, new_row], ignore_index=True)
        
        return sintactic_similarity_df


    def _compute_intraquestionnaire_syntactic_similarity(self, project_root, pred):
        try:
            generated = TFQuestionnairesDataset.from_json(project_root=project_root, questionnaire_id=self.questionnaire_id, json_data=pred["PREDICTED_JSON"])
            bleu_score = ExperimentEvaluator._compute_bleu_intraquestionnaire(generated.questions["NAME"])
            rouge_score = ExperimentEvaluator._compute_rouge_intraquestionnaire(generated.questions["NAME"])

            return pd.DataFrame({
                "QUESTIONNAIRE_ID": [self.questionnaire_id],
                "BLEU_SCORE": [bleu_score],
                "ROUGE_L_PRECISION": [rouge_score["precision"]],
                "ROUGE_L_RECALL": [rouge_score["recall"]],
                "ROUGE_L_F1": [rouge_score["f1"]]
            })                
        
        except Exception as e:
            return


    def _compute_bleu_intraquestionnaire(questions):
        scores = []
        for candidate in questions:
            for reference in questions:
                if (reference == candidate):
                    continue

                ref = reference.split()
                can = candidate.split()
                score = sentence_bleu([ref], can, smoothing_function=SmoothingFunction().method4, weights=ExperimentEvaluator.SYNT_SIM_BLEU_SCORE_N_GRAMS_WEIGHTS)
                scores.append(score)
        
        return np.mean(scores)
    

    def _compute_rouge_intraquestionnaire(questions):
        precisions = []
        recalls = []
        f1measures = []

        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

        for candidate in questions:
            for ref in questions:
                if (ref == candidate):
                    continue
                
                score = scorer.score(ref, candidate)
                precisions.append(score["rougeL"].precision)
                recalls.append(score["rougeL"].recall)
                f1measures.append(score["rougeL"].fmeasure)

        return {
            "precision": np.mean(precisions),
            "recall": np.mean(recalls),
            "f1": np.mean(f1measures)
        }


    def _compute_syntactic_similarity_with_samples(self, project_root, dataset):
        sintactic_similarity_df = pd.DataFrame(columns=self.SYNTACTIC_SIMILARITY_SAMPLE_COLUMNS)
        
        for pred in self.predictions.iterrows():
            self.set_questionnaire_id(pred[1]["QUESTIONNAIRE_ID"])
            sample_ids = pred[1]["SAMPLE_QUESTIONNAIRES_IDS"]

            sintactic_similarity_df = pd.concat([sintactic_similarity_df,
                                                 self._compute_syntactic_similarity_with_sample(project_root=project_root, dataset=dataset, sample_ids=sample_ids, generated_json=pred[1]["PREDICTED_JSON"])
                                                ], ignore_index=True)
        
        return sintactic_similarity_df
    

    def _compute_syntactic_similarity_with_sample(self, project_root, dataset, sample_ids, generated_json):
        try:
            sintactic_similarity_df = pd.DataFrame(columns=self.SYNTACTIC_SIMILARITY_SAMPLE_COLUMNS)

            generated = TFQuestionnairesDataset.from_json(project_root=project_root, questionnaire_id=self.questionnaire_id, json_data=generated_json)

            for sample_id in sample_ids:
                sample_questionnaire = dataset.get_questionnaire_data(sample_id)

                mean_bleu = ExperimentEvaluator._compute_bleu_with_sample(generated.questions["NAME"], sample_questionnaire.questions["NAME"])
                mean_rougeL = ExperimentEvaluator._compute_rouge_with_sample(generated.questions["NAME"], sample_questionnaire.questions["NAME"])

                new_row = pd.DataFrame({
                    "QUESTIONNAIRE_ID": [self.questionnaire_id],
                    "SAMPLE_QUESTIONNAIRE_ID": [sample_id],
                    "BLEU_SCORE": [mean_bleu],
                    "ROUGE_L_PRECISION": [mean_rougeL["precision"]],
                    "ROUGE_L_RECALL": [mean_rougeL["recall"]],
                    "ROUGE_L_F1": [mean_rougeL["f1"]]
                })
                sintactic_similarity_df = pd.concat([sintactic_similarity_df, new_row], ignore_index=True)
        except Exception as e:
            return
        
        return sintactic_similarity_df


    def _compute_bleu_with_sample(generated_questions, sample_questions):
        scores = []
        for candidate in generated_questions:
            for reference in sample_questions:
                ref = reference.split()
                can = candidate.split()
                score = sentence_bleu([ref], can, smoothing_function=SmoothingFunction().method4, weights=ExperimentEvaluator.SYNT_SIM_BLEU_SCORE_N_GRAMS_WEIGHTS)
                scores.append(score)
        
        return np.mean(scores)
    

    def _compute_rouge_with_sample(generated_questions, sample_questions):
        precisions = []
        recalls = []
        f1measures = []

        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

        for candidate in generated_questions:
            for ref in sample_questions:
                score = scorer.score(ref, candidate)
                precisions.append(score["rougeL"].precision)
                recalls.append(score["rougeL"].recall)
                f1measures.append(score["rougeL"].fmeasure)

        return {
            "precision": np.mean(precisions),
            "recall": np.mean(recalls),
            "f1": np.mean(f1measures)
        }
