import pandas as pd
import os
import numpy as np

class ModelEvaluator:
    # ------------
    # Constants
    # ------------
    EXPERIMENT_METRICS_COLUMNS = ["EXPERIMENT_ID", "ASW_BLEU_SCORE", "ASW_ROUGE_L_F1_SCORE", "QST_BLEU_SCORE", "QST_ROUGE_L_F1_SCORE"]
    EXPERIMENT_METRICS_FILENAME = "metrics.csv"

    EXPERIMENT_ERRORS_COLUMNS = ["EXPERIMENT_ID", 
                                 "CONVERSION_ERROR_RATE", "JSON_ERROR_RATE", "UNIQUE_QST_CODE_ERROR_RATE", "QST_WITHOUT_ANSWER_ERROR_RATE",
                                 "AVERAGE_QUESTION_NUMBER_DEVIATION", "AVERAGE_ANSWER_NUMBER_DEVIATION",
                                 "AVERAGE_QUESTION_NUMBER_DEVIATION_WITH_SAMPLE", "AVERAGE_ANSWER_NUMBER_DEVIATION_WITH_SAMPLE"]
    EXPERIMENT_ERRORS_FILENAME = "errors.csv"

    EXPERIMENT_SYNTACTIC_SIMILARITY_COLUMNS = ["EXPERIMENT_ID",
                                               "INTRAQSTN_BLEU_SCORE", "INTRAQSTN_ROUGE_L_F1_SCORE",
                                               "BLEU_SCORE_WITH_SAMPLE", "ROUGE_L_F1_SCORE_WITH_SAMPLE"]
    EXPERIMENT_SYNTACTIC_SIMILARITY_FILENAME = "syntactic_similarity.csv"


    # ------------
    # Constructor
    # ------------


    # ------------
    # Methods
    # ------------