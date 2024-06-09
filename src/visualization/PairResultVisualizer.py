import plotly.graph_objects as go
import pandas as pd
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
    def __init__(self, pair_dict):
        self.exp_pair_id = pair_dict["id"]
        self.exp_zero_shot = pair_dict["zero-shot"]
        self.exp_one_shot = pair_dict["one-shot"]
        self.exp_zero_shot_full = pair_dict["zero-shot_full"]
        self.exp_one_shot_full = pair_dict["one-shot_full"]

        self.ground_truth_questions = pd.DataFrame()
        self.zero_shot_questions = pd.DataFrame()
        self.one_shot_questions = pd.DataFrame()
        self.zero_shot_full_questions = pd.DataFrame()
        self.one_shot_full_questions = pd.DataFrame()


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
            qst_ref = zero_shot_preds[zero_shot_preds['QUESTIONNAIRE_ID'] == qst_id]["GROUND_TRUTH_JSON"].values[0]
            qst_0s_json = zero_shot_preds[zero_shot_preds['QUESTIONNAIRE_ID'] == qst_id]["PREDICTED_JSON"].values[0]
            qst_1s_json = one_shot_preds[one_shot_preds['QUESTIONNAIRE_ID'] == qst_id]["PREDICTED_JSON"].values[0]
            qst_0s_full_json = zero_shot_full_preds[zero_shot_full_preds['QUESTIONNAIRE_ID'] == qst_id]["PREDICTED_JSON"].values[0]
            qst_1s_full_json = one_shot_full_preds[one_shot_full_preds['QUESTIONNAIRE_ID'] == qst_id]["PREDICTED_JSON"].values[0]

            qst_ref = TFQuestionnairesDataset.from_json(project_root=os.path.abspath(os.path.join(os.getcwd(), os.pardir)), json_data=qst_ref, questionnaire_id=qst_id)
            qst_0s = TFQuestionnairesDataset.from_json(project_root=os.path.abspath(os.path.join(os.getcwd(), os.pardir)), json_data=qst_0s_json, questionnaire_id=qst_id)
            qst_1s = TFQuestionnairesDataset.from_json(project_root=os.path.abspath(os.path.join(os.getcwd(), os.pardir)), json_data=qst_1s_json, questionnaire_id=qst_id)
            qst_0s_full = TFQuestionnairesDataset.from_json(project_root=os.path.abspath(os.path.join(os.getcwd(), os.pardir)), json_data=qst_0s_full_json, questionnaire_id=qst_id)
            qst_1s_full = TFQuestionnairesDataset.from_json(project_root=os.path.abspath(os.path.join(os.getcwd(), os.pardir)), json_data=qst_1s_full_json, questionnaire_id=qst_id)

            self.ground_truth_questions = pd.concat([self.ground_truth_questions, qst_ref.questions], ignore_index=True)
            self.zero_shot_questions = pd.concat([self.zero_shot_questions, qst_0s.questions], ignore_index=True)
            self.one_shot_questions = pd.concat([self.one_shot_questions, qst_1s.questions], ignore_index=True)
            self.zero_shot_full_questions = pd.concat([self.zero_shot_full_questions, qst_0s_full.questions], ignore_index=True)
            self.one_shot_full_questions = pd.concat([self.one_shot_full_questions, qst_1s_full.questions], ignore_index=True)
            


    # ------------
    # Methods
    # ------------
