import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import os
import sys
sys.path.append('\\'.join(os.getcwd().split('\\')[:-1])+'\\src')


class GlobalResultVisualizer:
    # ------------
    # Constants
    # ------------
    ERRORS_FILENAME = "errors.csv"
    SYNTACTIC_SIMILARITY_FILENAME = "syntactic_similarity.csv"
    SEMANTIC_METRICS_FILENAME = "semantic_metrics.csv"
    SERENDIPITY_FILENAME = "serendipity_scores.csv"
    QST_TYPE_VARIABILITY = "qst_type_variability.csv"

    TF_PRIMARY_COLOR_1 = "#D0006F"
    TF_PRIMARY_COLOR_2 = "#24135F"

    TEMPERATURE_VALUES = [0, 0.25, 0.5]
    FREQUENCY_PENALTY_VALUES = [0, 0.5, 1]
    NOT_PERFORMED = [[1, 0.25], [0.5, 0.5]]

    # ------------
    # Constructor
    # ------------
    def __init__(self, project_root):
        result_dir = os.path.join(project_root, "results")

        self.intraquestionnaire_similarity = pd.read_csv(os.path.join(result_dir, self.SYNTACTIC_SIMILARITY_FILENAME))
        self.semantic_metrics = pd.read_csv(os.path.join(result_dir, self.SEMANTIC_METRICS_FILENAME))
        self.serendipity_scores = pd.read_csv(os.path.join(result_dir, self.SERENDIPITY_FILENAME))
        self.qst_type_variability = pd.read_csv(os.path.join(result_dir, self.QST_TYPE_VARIABILITY))
        self.errors = pd.read_csv(os.path.join(result_dir, self.ERRORS_FILENAME))

        self.experiment_names = self.errors.sort_values("EXPERIMENT_ID")["EXPERIMENT_ID"].unique()

        self.all_results = pd.DataFrame(columns=["EXPERIMENT_ID", "CONVERSION_ERROR_RATE", "JSON_ERROR_RATE", "INTRAQSTN_SIMILARITY", "SEMANTIC_SIMILARITY", "SERENDIPITY", "QST_TYPE_VARIABILITY"])
        self.all_results["EXPERIMENT_ID"] = self.experiment_names
        self.all_results["CONVERSION_ERROR_RATE"] = self.errors.sort_values("EXPERIMENT_ID")["CONVERSION_ERROR_RATE"]
        self.all_results["JSON_ERROR_RATE"] = self.errors.sort_values("EXPERIMENT_ID")["JSON_ERROR_RATE"]
        self.all_results["INTRAQSTN_SIMILARITY"] = self.intraquestionnaire_similarity.sort_values("EXPERIMENT_ID")["INTRAQSTN_ROUGE_L_F1_SCORE"]
        self.all_results["SEMANTIC_SIMILARITY"] = self.semantic_metrics.sort_values("EXPERIMENT_ID")["FINAL_SCORE"]
        self.all_results["SERENDIPITY"] = self.serendipity_scores.sort_values("EXPERIMENT_ID")["AVG_SERENDIPITY_SCORE"]
        self.all_results["QST_TYPE_VARIABILITY"] = self.qst_type_variability.sort_values("EXPERIMENT_ID")["AVG_QST_TYPE_VARIABILITY"]

    # ------------
    # Methods
    # ------------
    def get_best_and_worst_experiments(self):
        bw_conversion_errors_df = GlobalResultVisualizer._get_best_and_worst(self.errors, "CONVERSION_ERROR_RATE")
        bw_json_errors_df = GlobalResultVisualizer._get_best_and_worst(self.errors, "JSON_ERROR_RATE")
        bw_intraquestionnaire_similarity_df = GlobalResultVisualizer._get_best_and_worst(self.intraquestionnaire_similarity, "INTRAQSTN_ROUGE_L_F1_SCORE")
        bw_semantic_metrics_df = GlobalResultVisualizer._get_best_and_worst(self.semantic_metrics, "FINAL_SCORE")
        bw_serendipity_scores_df = GlobalResultVisualizer._get_best_and_worst(self.serendipity_scores, "AVG_SERENDIPITY_SCORE")
        bw_qst_type_variability_df = GlobalResultVisualizer._get_best_and_worst(self.qst_type_variability, "AVG_QST_TYPE_VARIABILITY")

        return bw_conversion_errors_df, bw_json_errors_df, bw_intraquestionnaire_similarity_df, bw_semantic_metrics_df, bw_serendipity_scores_df, bw_qst_type_variability_df


    def _get_best_and_worst(df, column_to_compare):
        best = df[df[column_to_compare] == df[column_to_compare].max()]
        worst = df[df[column_to_compare] == df[column_to_compare].min()]

        bw_errors_df = pd.concat([best, worst], ignore_index=True)

        return bw_errors_df
    

    def plot_errors_heatmap(self):
        self._plot_heatmap_by_t_fp(self.errors, "CONVERSION_ERROR_RATE", "Conversion error-rate", cmap="RdYlGn_r")
        self._plot_heatmap_by_t_fp(self.errors, "JSON_ERROR_RATE", "JSON error-rate", cmap="RdYlGn_r")


    def plot_qst_type_variabiliy_heatmap(self):
        self._plot_heatmap_by_t_fp(self.qst_type_variability, "AVG_QST_TYPE_VARIABILITY", "Question Type Variability", cmap="RdYlGn")


    def plot_intraquestionnaire_similarity_heatmap(self):
        self._plot_heatmap_by_t_fp(self.intraquestionnaire_similarity, "INTRAQSTN_ROUGE_L_F1_SCORE", "Intra-questionnaire similarity", cmap="RdYlGn_r")

    
    def plot_semantic_similarity_heatmap(self):
        self._plot_heatmap_by_t_fp(self.semantic_metrics, "FINAL_SCORE", "Semantic Similarity", cmap="RdYlGn")

    
    def plot_serendipity_heatmap(self):
        self._plot_heatmap_by_t_fp(self.serendipity_scores, "AVG_SERENDIPITY_SCORE", "Serendipity", cmap="RdYlGn")
        
    
    def _plot_heatmap_by_t_fp(self, df, column, title, cmap="RdYlGn", model="", has_full_params=-1, technique=""):
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
        
        for fp, t in self.NOT_PERFORMED:
            FP_idx = self.FREQUENCY_PENALTY_VALUES.index(fp)
            T_idx = self.TEMPERATURE_VALUES.index(t)
            metrics_matrix[FP_idx, T_idx] = np.nan

        for (fp, t), metrics in metrics_dict.items():
            if metrics:
                FP_idx = self.FREQUENCY_PENALTY_VALUES.index(fp)
                T_idx = self.TEMPERATURE_VALUES.index(t)
                metrics_matrix[FP_idx, T_idx] = np.mean(metrics)

        plt.figure(figsize=(8, 6))     # For a bigger plot -> plt.figure(figsize=(12, 9))
        
        plt.imshow(metrics_matrix, extent=[min(self.TEMPERATURE_VALUES), max(self.TEMPERATURE_VALUES), min(self.FREQUENCY_PENALTY_VALUES), max(self.FREQUENCY_PENALTY_VALUES)],
                   origin='lower', aspect='auto', cmap=cmap)
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=12)
        plt.xlabel("Temperature", fontsize=20)
        plt.ylabel("Frequency Penalty", fontsize=20)
        plt.title(title, fontsize=20)
        plt.xticks(self.TEMPERATURE_VALUES, fontsize=12)
        plt.yticks(self.FREQUENCY_PENALTY_VALUES, fontsize=12)

        # -------------
        # Uncomment to save the plot as PDF
        # -------------
        # pdf_filename = f"{title}.pdf"
        # plt.savefig(pdf_filename, format='pdf', bbox_inches='tight')

        plt.show()


    def plot_general_correlation_matrix(self):
        GlobalResultVisualizer._plot_correlation_matric(self.all_results.loc[:, self.all_results.columns != 'EXPERIMENT_ID'])


    def plot_correlation_matrix_by_config(self, title, model="", has_full_params=-1, shots=-1, temperature=-1, frequency_penalty=-1):
        df = self.all_results.copy()

        if model.__contains__("gpt-35-turbo"):
            df = df[df["EXPERIMENT_ID"].str.contains("gpt-35-turbo")]
        elif model.__contains__("gpt-4"):
            df = df[df["EXPERIMENT_ID"].str.contains("gpt-4")]

        if has_full_params == 1:
            df = df[df["EXPERIMENT_ID"].str.contains("FULL")]
        elif has_full_params == 0:
            df = df[~df["EXPERIMENT_ID"].str.contains("FULL")]

        if shots == 0:
            df = df[df["EXPERIMENT_ID"].str.contains("0s")]
        elif shots == 1:
            df = df[df["EXPERIMENT_ID"].str.contains("1s")]

        if temperature == 0:
            df = df[df["EXPERIMENT_ID"].str.contains("0T")]
        elif temperature == 0.25:
            df = df[df["EXPERIMENT_ID"].str.contains("0.25T")]
        elif temperature == 0.5:
            df = df[df["EXPERIMENT_ID"].str.contains("0.5T")]

        if frequency_penalty == 0:
            df = df[df["EXPERIMENT_ID"].str.contains("0FP")]
        elif frequency_penalty == 0.5:
            df = df[df["EXPERIMENT_ID"].str.contains("0.5FP")]
        elif frequency_penalty == 1:
            df = df[df["EXPERIMENT_ID"].str.contains("1FP")]

        GlobalResultVisualizer._plot_correlation_matric(df.loc[:, df.columns != 'EXPERIMENT_ID'], title)



    def _plot_correlation_matric(df, title=""):
        matrix = df.corr()

        plt.imshow(matrix, cmap='BuPu')
        plt.colorbar()

        variables = []
        for i in matrix.columns:
            variables.append(i)

        plt.xticks(range(len(matrix)), variables, rotation=45, ha='right')
        plt.yticks(range(len(matrix)), variables)

        if title:
            plt.title(title)

        plt.show()
