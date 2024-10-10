import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import os
import sys
sys.path.append('\\'.join(os.getcwd().split('\\')[:-1])+'\\src')
from src.visualization.ModelResultVisualizer import ModelResultVisualizer


class GlobalResultVisualizer:
    # ------------
    # Constants
    # ------------
    TESTED_MODELS = ["GPT", "LLaMa", "Mistral"]

    TF_PRIMARY_COLOR_1 = "#D0006F"
    TF_PRIMARY_COLOR_2 = "#24135F"

    TEMPERATURE_VALUES = [0, 0.25, 0.5]
    FREQUENCY_PENALTY_VALUES = [0, 0.5, 1]
    NOT_PERFORMED = [[1, 0.25], [0.5, 0.5]]

    # ------------
    # Constructor
    # ------------
    def __init__(self, project_root, task="Survey", prompt_version="1.0"):

        self.gpt_results = ModelResultVisualizer(project_root, task=task, prompt_version=prompt_version, model="GPT")
        self.llama_results = ModelResultVisualizer(project_root, task=task, prompt_version=prompt_version, model= "LLaMa")
        self.mistral_results = ModelResultVisualizer(project_root, task=task, prompt_version=prompt_version, model="Mistral")

    # ------------
    # Methods
    # ------------
    def plot_errors_heatmap(self):
        df_list = [self.gpt_results.errors, self.llama_results.errors, self.mistral_results.errors]
        self._plot_heatmap_by_t_fp(df_list, "CONVERSION_ERROR_RATE", "Conversion error-rate", cmap="RdYlGn_r")
        self._plot_heatmap_by_t_fp(df_list, "JSON_ERROR_RATE", "JSON error-rate", cmap="RdYlGn_r")


    def plot_qst_type_variabiliy_heatmap(self):
        df_list = [self.gpt_results.qst_type_variability, self.llama_results.qst_type_variability, self.mistral_results.qst_type_variability]
        self._plot_heatmap_by_t_fp(df_list, "AVG_QST_TYPE_VARIABILITY", "Question type variability", cmap="RdYlGn")


    def plot_intraquestionnaire_similarity_heatmap(self):
        df_list = [self.gpt_results.intraquestionnaire_similarity, self.llama_results.intraquestionnaire_similarity, self.mistral_results.intraquestionnaire_similarity]
        self._plot_heatmap_by_t_fp(df_list, "INTRAQSTN_ROUGE_L_F1_SCORE", "Intraquestionnaire similarity", cmap="RdYlGn_r")

    
    def plot_semantic_similarity_heatmap(self):
        df_list = [self.gpt_results.semantic_metrics, self.llama_results.semantic_metrics, self.mistral_results.semantic_metrics]
        self._plot_heatmap_by_t_fp(df_list, "FINAL_SCORE", "Semantic Similarity", cmap="RdYlGn")

    
    def plot_serendipity_heatmap(self):
        df_list = [self.gpt_results.serendipity_scores, self.llama_results.serendipity_scores, self.mistral_results.serendipity_scores]
        self._plot_heatmap_by_t_fp(df_list, "AVG_SERENDIPITY_SCORE", "Serendipity", cmap="RdYlGn")
        

    def _plot_heatmap_by_t_fp(self, df_list, column, title, cmap="RdYlGn", model="", has_full_params=-1, technique=""):
        k = len(df_list)
        fig, axes = plt.subplots(1, k, figsize=(24, 6))

        for i in range(k):
            metric_matrix = ModelResultVisualizer.compute_metrics_matrix(df_list[i], column, model, has_full_params, technique)

            im = axes[i].imshow(metric_matrix, extent=[min(self.TEMPERATURE_VALUES), max(self.TEMPERATURE_VALUES), min(self.FREQUENCY_PENALTY_VALUES), max(self.FREQUENCY_PENALTY_VALUES)],
                        origin='lower', aspect='auto', cmap=cmap)

            axes[i].set_xlabel("Temperature", fontsize=14)
            axes[i].set_ylabel("Frequency Penalty", fontsize=14)
            axes[i].set_title(self.TESTED_MODELS[i], fontsize=16)
            axes[i].set_xticks(self.TEMPERATURE_VALUES)
            axes[i].set_yticks(self.FREQUENCY_PENALTY_VALUES)

            cbar = fig.colorbar(im, ax=axes[i])
            cbar.ax.tick_params(labelsize=10)

        fig.suptitle(title, fontsize=20)
        plt.tight_layout()
        # -------------
        # Uncomment to save the plot as PDF
        # -------------
        # pdf_filename = f"{title}.pdf"
        # plt.savefig(pdf_filename, format='pdf', bbox_inches='tight')

        plt.show()
