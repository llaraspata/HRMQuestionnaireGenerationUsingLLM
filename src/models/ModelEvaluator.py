import pandas as pd
import os
import numpy as np

class ModelEvaluator:
    # ------------
    # Constants
    # ------------
    SYNTACTIC_METRICS_COLUMNS = ["EXPERIMENT_ID", "ASW_BLEU_SCORE", "ASW_ROUGE_L_F1_SCORE", "QST_BLEU_SCORE", "QST_ROUGE_L_F1_SCORE"]
    SYNTACTIC_METRICS_FILENAME = "syntactic_metrics.csv"

    ERRORS_COLUMNS = ["EXPERIMENT_ID",
                      "CONVERSION_ERROR_RATE", "JSON_ERROR_RATE", 
                      "NOT_SINGLE_QUESTIONNAIRE_ERROR_RATE", "AVG_GENERATED_QUESTIONNAIRES",
                      "UNIQUE_QST_CODE_ERROR_RATE", "AVG_NOT_UNIQUE_QST_CODE",
                      "QST_WITHOUT_ANSWER_ERROR_RATE", "AVG_QST_WITHOUT_ANSWER",
                      "AVERAGE_QUESTION_NUMBER_DEVIATION", "AVERAGE_ANSWER_NUMBER_DEVIATION",
                      "AVERAGE_QUESTION_NUMBER_DEVIATION_WITH_SAMPLE"]
    ERRORS_FILENAME = "errors.csv"

    SYNTACTIC_SIMILARITY_COLUMNS = ["EXPERIMENT_ID",
                                    "INTRAQSTN_BLEU_SCORE", "INTRAQSTN_ROUGE_L_F1_SCORE",
                                    "BLEU_SCORE_WITH_SAMPLE", "ROUGE_L_F1_SCORE_WITH_SAMPLE"]
    SYNTACTIC_SIMILARITY_FILENAME = "syntactic_similarity.csv"

    SEMANTIC_METRICS_COLUMNS = ["EXPERIMENT_ID", "FINAL_SCORE", "SIMILARITY_WITH_QUESTION", "SIMILARITY_WITH_TOPIC", "POSITION_DEVIATION"]
    SEMANTIC_METRICS_FILENAME = "semantic_metrics.csv"


    # ------------
    # Constructor
    # ------------
    def __init__(self, results_dir):
        self.results_dir = results_dir
        self.errors = pd.DataFrame(columns=self.ERRORS_COLUMNS)
        self.syntactic_metrics = pd.DataFrame(columns=self.SYNTACTIC_METRICS_COLUMNS)
        self.syntactic_symilarity = pd.DataFrame(columns=self.SYNTACTIC_SIMILARITY_COLUMNS)
        self.semantic_metrics = pd.DataFrame(columns=self.SEMANTIC_METRICS_COLUMNS)


    # ------------
    # Methods
    # ------------
    def evaluate(self):
        for subfolder in os.listdir(self.results_dir):
            experiment_path = os.path.join(self.results_dir, subfolder)

            if os.path.isdir(experiment_path):
                # 1. Syntactic metrics
                self._evaluate_syntactic_metrics(subfolder, experiment_path)

                # 2. Semantic metrics
                self._evaluate_semantic_metrics(subfolder, experiment_path)

                # 3. Syntactic similarity
                self._evaluate_syntactic_similarity(subfolder, experiment_path)
                
                # 4. Errors
                self._evaluate_errors(subfolder, experiment_path)

        self.syntactic_metrics.to_csv(os.path.join(self.results_dir, self.SYNTACTIC_METRICS_FILENAME))
        self.semantic_metrics.to_csv(os.path.join(self.results_dir, self.SEMANTIC_METRICS_FILENAME))
        self.syntactic_symilarity.to_csv(os.path.join(self.results_dir, self.SYNTACTIC_SIMILARITY_FILENAME))
        self.errors.to_csv(os.path.join(self.results_dir, self.ERRORS_FILENAME))

    
    def _evaluate_syntactic_metrics(self, id, experiment_path):
        bleu_scores = pd.read_csv(os.path.join(experiment_path, "BLEU_questionnaires.csv"))
        rouge_scores = pd.read_csv(os.path.join(experiment_path, "ROUGE_L_F1_questionnaires.csv"))

        exp_scores = pd.DataFrame({
            "EXPERIMENT_ID": [id],
            "ASW_BLEU_SCORE": np.mean(bleu_scores["ASW_BLEU_SCORE_MEAN"]),
            "ASW_ROUGE_L_F1_SCORE": np.mean(rouge_scores["ASW_ROUGE_SCORE_MEAN"]),
            "QST_BLEU_SCORE": np.mean(bleu_scores["QST_BLEU_SCORE_MEAN"]),
            "QST_ROUGE_L_F1_SCORE": np.mean(rouge_scores["QST_ROUGE_SCORE_MEAN"])
        })

        self.syntactic_metrics = pd.concat([self.syntactic_metrics, exp_scores], ignore_index=True)

    
    def _evaluate_semantic_metrics(self, id, experiment_path):
        scores = pd.read_csv(os.path.join(experiment_path, "SemanticSimilarity_questionnaires.csv"))

        exp_scores = pd.DataFrame({
            "EXPERIMENT_ID": [id],
            "FINAL_SCORE": np.mean(scores["FINAL_SCORE_MEAN"]),
            "SIMILARITY_WITH_QUESTION": np.mean(scores["SIMILARITY_WITH_QUESTION_MEAN"]),
            "SIMILARITY_WITH_TOPIC": np.mean(scores["SIMILARITY_WITH_TOPIC_MEAN"]),
            "POSITION_DEVIATION": np.mean(scores["POSITION_DEVIATION"])
        })

        self.semantic_metrics = pd.concat([self.semantic_metrics, exp_scores], ignore_index=True)


    def _evaluate_syntactic_similarity(self, id, experiment_path):
        intraqstn_scores = pd.read_csv(os.path.join(experiment_path, "Intraquestionnaire_Syntactic_Similarity.csv"))

        if id.startswith("1s") :
            with_sample_scores = pd.read_csv(os.path.join(experiment_path, "Syntactic_Similarity_With_Sample.csv"))
            exp_scores = pd.DataFrame({
                "EXPERIMENT_ID": [id],
                "INTRAQSTN_BLEU_SCORE": np.mean(intraqstn_scores["QST_BLEU_SCORE"]),
                "INTRAQSTN_ROUGE_L_F1_SCORE": np.mean(intraqstn_scores["QST_ROUGE_L_F1_SCORE"]),
                "BLEU_SCORE_WITH_SAMPLE": np.mean(with_sample_scores["BLEU_SCORE"]),
                "ROUGE_L_F1_SCORE_WITH_SAMPLE": np.mean(with_sample_scores["ROUGE_L_F1"])
            })
        else:
            exp_scores = pd.DataFrame({
                "EXPERIMENT_ID": [id],
                "INTRAQSTN_BLEU_SCORE": np.mean(intraqstn_scores["QST_BLEU_SCORE"]),
                "INTRAQSTN_ROUGE_L_F1_SCORE": np.mean(intraqstn_scores["QST_ROUGE_L_F1_SCORE"]),
            })
        
        self.syntactic_symilarity = pd.concat([self.syntactic_symilarity, exp_scores], ignore_index=True)


    def _evaluate_errors(self, id, experiment_path):
        stats = pd.read_csv(os.path.join(experiment_path, "statistics.csv"))

        samples = len(stats["QUESTIONNAIRE_ID"])
        succeeded_generations = samples - stats[stats["ERROR_MESSAGE"] is not None].count()
        generated_questions = stats[stats["GENERATED_QUESTION_NUMBER"] >= 0]["GENERATED_QUESTION_NUMBER"].sum()

        conversion_rate = (100 * stats[stats["CONVERSION_ERROR"]].count()) / samples
        json_rate = (100 * stats[stats["IS_JSON"]].count()) / samples

        not_one_qstn_rate = (100 * stats[stats["GENERATED_QUESTIONNAIRES"] > 1].count()) / succeeded_generations
        unique_qst_code_rate = (100 * stats[stats["NOT_UNIOUE_QUESTION_CODES"] > 0].count()) / generated_questions
        qst_no_asw_rate = (100 * stats[stats["QUESTIONS_WITH_MISSING_ANSWERS"] > 0].count()) / generated_questions

        avg_not_unique_qst_code = stats[stats["NOT_UNIOUE_QUESTION_CODES"] > 0]["NOT_UNIOUE_QUESTION_CODES"].mean()
        avg_qst_no_asw = stats[stats["QUESTIONS_WITH_MISSING_ANSWERS"] > 0]["QUESTIONS_WITH_MISSING_ANSWERS"].mean()
        avg_generated_qstn = stats[stats["GENERATED_QUESTIONNAIRES"] >= 0]["GENERATED_QUESTIONNAIRES"].mean()

        avg_qst_dev = stats[stats["QUESTION_NUMBER_DEVIATION"] > 0]["QUESTION_NUMBER_DEVIATION"].mean()
        avg_asw_dev = stats[stats["AVERAGE_ANSWER_NUMBER_DEVIATION"] > 0]["AVERAGE_ANSWER_NUMBER_DEVIATION"].mean()

        if id.startswith("1s"):
            avg_qst_sample_dev = stats[stats["QUESTION_NUMBER_DEVIATION_FROM_SAMPLE"] > 0]["QUESTION_NUMBER_DEVIATION_FROM_SAMPLE"].mean()

            exp_stats = pd.DataFrame({
                "EXPERIMENT_ID": [id],
                "CONVERSION_ERROR_RATE": conversion_rate,
                "JSON_ERROR_RATE": json_rate,
                "NOT_SINGLE_QUESTIONNAIRE_ERROR_RATE": not_one_qstn_rate,
                "AVG_GENERATED_QUESTIONNAIRES": avg_generated_qstn,
                "UNIQUE_QST_CODE_ERROR_RATE": unique_qst_code_rate,
                "AVG_NOT_UNIQUE_QST_CODE": avg_not_unique_qst_code,
                "QST_WITHOUT_ANSWER_ERROR_RATE": qst_no_asw_rate,
                "AVG_QST_WITHOUT_ANSWER": avg_qst_no_asw,
                "AVERAGE_QUESTION_NUMBER_DEVIATION": avg_qst_dev,
                "AVERAGE_ANSWER_NUMBER_DEVIATION": avg_asw_dev,
                "AVERAGE_QUESTION_NUMBER_DEVIATION_WITH_SAMPLE": avg_qst_sample_dev
            })
        else:
            exp_stats = pd.DataFrame({
                "EXPERIMENT_ID": [id],
                "CONVERSION_ERROR_RATE": conversion_rate,
                "JSON_ERROR_RATE": json_rate,
                "NOT_SINGLE_QUESTIONNAIRE_ERROR_RATE": not_one_qstn_rate,
                "AVG_GENERATED_QUESTIONNAIRES": avg_generated_qstn,
                "UNIQUE_QST_CODE_ERROR_RATE": unique_qst_code_rate,
                "AVG_NOT_UNIQUE_QST_CODE": avg_not_unique_qst_code,
                "QST_WITHOUT_ANSWER_ERROR_RATE": qst_no_asw_rate,
                "AVG_QST_WITHOUT_ANSWER": avg_qst_no_asw,
                "AVERAGE_QUESTION_NUMBER_DEVIATION": avg_qst_dev,
                "AVERAGE_ANSWER_NUMBER_DEVIATION": avg_asw_dev
            })

        self.errors = pd.concat([self.errors, exp_stats], ignore_index=True)