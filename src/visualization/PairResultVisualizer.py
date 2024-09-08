import string
import plotly.graph_objects as go
import plotly.figure_factory as ff
import pandas as pd
import numpy as np
import os
import sys
sys.path.append('\\'.join(os.getcwd().split('\\')[:-1])+'\\src')

from src.data.TFQuestionnairesDataset import TFQuestionnairesDataset
import src.visualization.visualize as vis


class PairResultVisualizer:
    # ------------
    # Constants
    # ------------
    PREDICTION_FILENAME = "predictions.pkl"

    QUESTION_COLUMNS = ["ID", "TYPE_ID", "QUESTIONNAIRE_ID", "CODE", "NAME", "DISPLAY_ORDER"]




    # ------------
    # Constructor
    # ------------
    def __init__(self, pair_dict, project_root):
        self.exp_pair_id = pair_dict["id"]
        self.exp_zero_shot = pair_dict["zero-shot"]
        self.exp_one_shot = pair_dict["one-shot"]
        self.exp_zero_shot_full = pair_dict["zero-shot_full"]
        self.exp_one_shot_full = pair_dict["one-shot_full"]

        
        self.zero_shot_questions = pd.DataFrame()
        self.zero_shot_answers = pd.DataFrame()

        self.one_shot_questions = pd.DataFrame()
        self.one_shot_answers = pd.DataFrame()

        self.zero_shot_full_questions = pd.DataFrame()
        self.zero_shot_full_answers = pd.DataFrame()

        self.one_shot_full_questions = pd.DataFrame()
        self.one_shot_full_answers = pd.DataFrame()

        dataset = TFQuestionnairesDataset()
        dataset.load_data(project_root=project_root)
        self.ground_truth_questions = dataset.questions
        self.ground_truth_answers = dataset.answers


    def add_pair(self, pair_dict):
        self.exp_pair_id = pair_dict["id"]
        self.exp_zero_shot = pair_dict["zero-shot"]
        self.exp_one_shot = pair_dict["one-shot"]
        self.exp_zero_shot_full = pair_dict["zero-shot_full"]
        self.exp_one_shot_full = pair_dict["one-shot_full"]


    def load_data(self, project_root):
        predictions_dir = os.path.join(project_root, "models")
        zero_shot_dir = os.path.join(predictions_dir, self.exp_zero_shot)
        one_shot_dir = os.path.join(predictions_dir, self.exp_one_shot)
        zero_shot_full_dir = os.path.join(predictions_dir, self.exp_zero_shot_full)
        one_shot_full_dir = os.path.join(predictions_dir, self.exp_one_shot_full)

        zero_shot_preds = pd.read_pickle(os.path.join(zero_shot_dir, self.PREDICTION_FILENAME))
        one_shot_preds = pd.read_pickle(os.path.join(one_shot_dir, self.PREDICTION_FILENAME))
        zero_shot_full_preds = pd.read_pickle(os.path.join(zero_shot_full_dir, self.PREDICTION_FILENAME))
        one_shot_full_preds = pd.read_pickle(os.path.join(one_shot_full_dir, self.PREDICTION_FILENAME))

        for qst_id in zero_shot_preds["QUESTIONNAIRE_ID"].unique():
            qst_0s_json = zero_shot_preds[zero_shot_preds['QUESTIONNAIRE_ID'] == qst_id]["PREDICTED_JSON"].values[0]
            qst_1s_json = one_shot_preds[one_shot_preds['QUESTIONNAIRE_ID'] == qst_id]["PREDICTED_JSON"].values[0]
            qst_0s_full_json = zero_shot_full_preds[zero_shot_full_preds['QUESTIONNAIRE_ID'] == qst_id]["PREDICTED_JSON"].values[0]
            qst_1s_full_json = one_shot_full_preds[one_shot_full_preds['QUESTIONNAIRE_ID'] == qst_id]["PREDICTED_JSON"].values[0]
            
            try:
                qst_0s = TFQuestionnairesDataset.from_json(project_root=os.path.abspath(os.path.join(os.getcwd(), os.pardir)), json_data=qst_0s_json, questionnaire_id=qst_id)
                self.zero_shot_questions = pd.concat([self.zero_shot_questions, qst_0s.questions], ignore_index=True)
                self.zero_shot_answers = pd.concat([self.zero_shot_answers, qst_0s.answers], ignore_index=True)
            except:
                a = 0
            
            try:
                qst_1s = TFQuestionnairesDataset.from_json(project_root=os.path.abspath(os.path.join(os.getcwd(), os.pardir)), json_data=qst_1s_json, questionnaire_id=qst_id)
                self.one_shot_questions = pd.concat([self.one_shot_questions, qst_1s.questions], ignore_index=True)
                self.one_shot_answers = pd.concat([self.one_shot_answers, qst_1s.answers], ignore_index=True)
            except:
                a = 0
            
            try:
                qst_0s_full = TFQuestionnairesDataset.from_json(project_root=os.path.abspath(os.path.join(os.getcwd(), os.pardir)), json_data=qst_0s_full_json, questionnaire_id=qst_id)
                self.zero_shot_full_questions = pd.concat([self.zero_shot_full_questions, qst_0s_full.questions], ignore_index=True)
                self.zero_shot_full_answers = pd.concat([self.zero_shot_full_answers, qst_0s_full.answers], ignore_index=True)
            except:
                a = 0
            
            try:
                qst_1s_full = TFQuestionnairesDataset.from_json(project_root=os.path.abspath(os.path.join(os.getcwd(), os.pardir)), json_data=qst_1s_full_json, questionnaire_id=qst_id)
                self.one_shot_full_questions = pd.concat([self.one_shot_full_questions, qst_1s_full.questions], ignore_index=True)
                self.one_shot_full_answers = pd.concat([self.one_shot_full_answers, qst_1s_full.answers], ignore_index=True)
            except:
                a = 0
            


    # ------------
    # Methods
    # ------------
    def count_words(sentence):
        """
        Counts the number of words in a sentence, ignoring punctuation.
        """
        try:
            sentence = sentence.translate(str.maketrans('', '', string.punctuation))
            splitted = sentence.split()
            return len(splitted)
        except:
            return 0
        
    def plot_length_distribution_comparison(df_0s, df_1s, column, name, element):
        """
            Plots the distribution of the length of the reviews for zero and one shot predictions.
        """
        lens_0s = [PairResultVisualizer.count_words(sentence) for sentence in df_0s[df_0s[column].notna()][column]]
        lens_1s = [PairResultVisualizer.count_words(sentence) for sentence in df_1s[df_1s[column].notna()][column]]
        
        # mean_lens_0s = np.mean(lens_0s)
        # max_lens_0s = np.max(lens_0s)
        # min_lens_0s = np.min(lens_0s)

        # mean_lens_1s = np.mean(lens_1s)
        # max_lens_1s = np.max(lens_1s)
        # min_lens_1s = np.min(lens_1s)

        # title = f"""{name} - {element} length distribution 
        # <br>
        # 0s -> [Mean: {int(round(mean_lens_0s))}, Max: {max_lens_0s}, Min: {min_lens_0s}]    1s -> [Mean: {int(round(mean_lens_1s))}, Max: {max_lens_1s}, Min: {min_lens_1s}]"""

        fig = go.Figure()

        fig.add_trace(go.Histogram(x=lens_0s, name="Zero-Shot", marker_color="#D0006F", histnorm='probability'))
        fig.add_trace(go.Histogram(x=lens_1s, name="One-Shot", marker_color="#24135F", histnorm='probability'))


        fig.update_layout(
            title=f"{name} - {element} length distribution",
            xaxis_title="Length",
            yaxis_title="Frequency"
        )

        fig.show()


    def plot_length_distribution_comparison_with_ref(df_ref, df_0s, df_1s, column, name, element):
        """
            Plots the distribution of the length of the reviews for zero and one shot predictions.
        """
        lens_ref = [PairResultVisualizer.count_words(sentence) for sentence in df_ref[df_ref[column].notna()][column]]
        lens_0s = [PairResultVisualizer.count_words(sentence) for sentence in df_0s[df_0s[column].notna()][column]]
        lens_1s = [PairResultVisualizer.count_words(sentence) for sentence in df_1s[df_1s[column].notna()][column]]

        # mean_lens_ref = np.mean(lens_ref)
        # max_lens_ref = np.max(lens_ref)
        # min_lens_ref = np.min(lens_ref)

        # mean_lens_0s = np.mean(lens_0s)
        # max_lens_0s = np.max(lens_0s)
        # min_lens_0s = np.min(lens_0s)

        # mean_lens_1s = np.mean(lens_1s)
        # max_lens_1s = np.max(lens_1s)
        # min_lens_1s = np.min(lens_1s)

        # title = f"""{name} - {element} length distribution 
        # <br>
        # GT -> [Mean: {int(round(mean_lens_ref))}, Max: {max_lens_ref}, Min: {min_lens_ref}]    0s -> [Mean: {int(round(mean_lens_0s))}, Max: {max_lens_0s}, Min: {min_lens_0s}]    1s -> [Mean: {int(round(mean_lens_1s))}, Max: {max_lens_1s}, Min: {min_lens_1s}]"""


        fig = go.Figure()

        fig.add_trace(go.Histogram(x=lens_ref, name="Human-written", marker_color="#89A9EE", histnorm='probability'))
        fig.add_trace(go.Histogram(x=lens_0s, name="Zero-Shot", marker_color="#D0006F", histnorm='probability'))
        fig.add_trace(go.Histogram(x=lens_1s, name="One-Shot", marker_color="#24135F", histnorm='probability'))


        fig.update_layout(
            title=f"{name} - {element} length distribution",
            xaxis_title="Length",
            yaxis_title="Frequency"
        )

        fig.show()