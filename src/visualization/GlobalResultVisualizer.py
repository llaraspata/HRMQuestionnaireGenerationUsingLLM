import plotly.graph_objects as go
import pandas as pd
import os
import sys
sys.path.append('\\'.join(os.getcwd().split('\\')[:-1])+'\\src')

import src.visualization.visualize as vis


class GlobalResultVisualizer:
    # ------------
    # Constants
    # ------------
    ERRORS_FILENAME = "errors.csv"
    SYNTACTIC_SIMILARITY_FILENAME = "syntactic_similarity.csv"
    SEMANTIC_METRICS_FILENAME = "semantic_metrics.csv"
    SERENDIPITY_FILENAME = "serendipity_scores.csv"

    TF_PRIMARY_COLOR_1 = "#D0006F"
    TF_PRIMARY_COLOR_2 = "#24135F"

    # ------------
    # Constructor
    # ------------
    def __init__(self, project_root):
        result_dir = os.path.join(project_root, "results")

        self.intraquestionnaire_similarity = pd.read_csv(os.path.join(result_dir, self.SYNTACTIC_SIMILARITY_FILENAME))
        self.semantic_metrics = pd.read_csv(os.path.join(result_dir, self.SEMANTIC_METRICS_FILENAME))
        self.serendipity_scores = pd.read_csv(os.path.join(result_dir, self.SERENDIPITY_FILENAME))
        self.errors = pd.read_csv(os.path.join(result_dir, self.ERRORS_FILENAME))
        

    # ------------
    # Methods
    # ------------
    def get_best_and_worst_experiments(self):
        bw_conversion_errors_df = GlobalResultVisualizer._get_best_and_worst(self.errors, "CONVERSION_ERROR_RATE")
        bw_json_errors_df = GlobalResultVisualizer._get_best_and_worst(self.errors, "JSON_ERROR_RATE")
        bw_intraquestionnaire_similarity_df = GlobalResultVisualizer._get_best_and_worst(self.intraquestionnaire_similarity, "INTRAQSTN_ROUGE_L_F1_SCORE")
        bw_semantic_metrics_df = GlobalResultVisualizer._get_best_and_worst(self.semantic_metrics, "FINAL_SCORE")
        bw_serendipity_scores_df = GlobalResultVisualizer._get_best_and_worst(self.serendipity_scores, "AVG_SERENDIPITY_SCORE")

        return bw_conversion_errors_df, bw_json_errors_df, bw_intraquestionnaire_similarity_df, bw_semantic_metrics_df, bw_serendipity_scores_df


    def _get_best_and_worst(df, column_to_compare):
        best = df[df[column_to_compare] == df[column_to_compare].max()]
        worst = df[df[column_to_compare] == df[column_to_compare].min()]

        bw_errors_df = pd.concat([best, worst], ignore_index=True)

        return bw_errors_df

    
