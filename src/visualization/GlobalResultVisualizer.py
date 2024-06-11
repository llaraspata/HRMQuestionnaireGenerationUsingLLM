import matplotlib.pyplot as plt
import numpy as np
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

    TEMPERATURE_VALUES = [0, 0.25, 0.5]
    FREQUENCY_PENALTY_VALUES = [0, 0.5, 1]

    # ------------
    # Constructor
    # ------------
    def __init__(self, project_root):
        result_dir = os.path.join(project_root, "results")

        self.intraquestionnaire_similarity = pd.read_csv(os.path.join(result_dir, self.SYNTACTIC_SIMILARITY_FILENAME))
        self.semantic_metrics = pd.read_csv(os.path.join(result_dir, self.SEMANTIC_METRICS_FILENAME))
        self.serendipity_scores = pd.read_csv(os.path.join(result_dir, self.SERENDIPITY_FILENAME))
        self.errors = pd.read_csv(os.path.join(result_dir, self.ERRORS_FILENAME))

        self.experiment_names = self.errors["EXPERIMENT_ID"].unique()
        

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
    

    def plot_errors_heatmap(self):
        self._plot_result_heatmap(self.errors, "CONVERSION_ERROR_RATE", "Conversion error-rate", cmap="RdYlGn_r")
        self._plot_result_heatmap(self.errors, "JSON_ERROR_RATE", "JSON error-rate", cmap="RdYlGn_r")
        
    
    def _plot_result_heatmap(self, df, column, title, cmap="RdYlGn", model="", has_full_params=-1, technique=""):
        metrics_dict = {(fp, t): [] for fp in self.FREQUENCY_PENALTY_VALUES for t in self.TEMPERATURE_VALUES}

        for _, row in df[df[column].notna()].iterrows():
            exp_id = row["EXPERIMENT_ID"]

            metric = row[column]

            if model and not exp_id.__contains__(model):
                continue

            if (has_full_params == 1 and not exp_id.__contains__("FULL")) or (has_full_params == 0 and exp_id.__contains__("FULL")):
                continue

            if technique and not exp_id.__contains__(technique):
                continue

            parts = exp_id.split('_')
            
            if parts.__contains__("JSON"):
                parts = parts[:-1]
            
            if exp_id.__contains__("FULL"):
                T_value = float(parts[4][:-1])
                FP_value = float(parts[5][:-2])
            else:
                T_value = float(parts[3][:-1])
                FP_value = float(parts[4][:-2])

            metrics_dict[(FP_value, T_value)].append(metric)

        metrics_matrix = np.zeros((len(self.FREQUENCY_PENALTY_VALUES), len(self.TEMPERATURE_VALUES)))
        
        not_performed = [[1, 0.25], [0.5, 0.5]]
        for fp, t in not_performed:
            FP_idx = self.FREQUENCY_PENALTY_VALUES.index(fp)
            T_idx = self.TEMPERATURE_VALUES.index(t)
            metrics_matrix[FP_idx, T_idx] = np.nan

        for (fp, t), metrics in metrics_dict.items():
            if metrics:
                FP_idx = self.FREQUENCY_PENALTY_VALUES.index(fp)
                T_idx = self.TEMPERATURE_VALUES.index(t)
                metrics_matrix[FP_idx, T_idx] = np.mean(metrics)

        plt.figure(figsize=(8, 6))
        plt.imshow(metrics_matrix, extent=[min(self.TEMPERATURE_VALUES), max(self.TEMPERATURE_VALUES), min(self.FREQUENCY_PENALTY_VALUES), max(self.FREQUENCY_PENALTY_VALUES)],
                   origin='lower', aspect='auto', cmap=cmap)
        plt.colorbar(label='Values')
        plt.xlabel("Temperature")
        plt.ylabel("Frequency Penalty")
        plt.title(title)
        plt.xticks(self.TEMPERATURE_VALUES)
        plt.yticks(self.FREQUENCY_PENALTY_VALUES)
        plt.show()

