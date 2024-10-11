import os
import pandas as pd
import numpy as np

class PredictionUserPrompt:
    ESSENTIAL_COLUMNS = ["CODE", "ORDER", "TASK", "PROMPT_PART"]

    TASK_FILENAMES = {
          "Survey": "survey_generation.csv"
    }

    def __init__(self, project_root, prompt_version="1.0", task="Survey"):
        self.prompt_version = prompt_version
        self.task = task
        self._set_prompts(project_root)


    def _set_prompts(self, project_root):
        prompts_path = os.path.join(project_root, "prompts", self.TASK_FILENAMES[self.task])
        prompts_df = pd.read_csv(prompts_path)
        
        prompts_df = prompts_df[prompts_df["TYPE"] == 'U']
        self.prompts_df = prompts_df[prompts_df["VERSION"] == np.float64(self.prompt_version)][self.ESSENTIAL_COLUMNS].sort_values(by="ORDER")

    
    def build_prompt(self, has_full_params=True, params=[], qst_types_df=None):
        prompt = ""

        for row in self.prompts_df.iterrows():

            if self.prompt_version != "2.0":
                if has_full_params and row[1]["TASK"] == "ONLY_TOPIC":
                    continue
                elif not has_full_params and row[1]["TASK"] == "FULL":
                    continue
            elif row[1]["TASK"] == "ONLY_TOPIC" or row[1]["TASK"] == "FULL":
                continue

            if row[1]["CODE"] == "SINGLE_QUESTION_TYPE_DESCRIPTION":
                for _, type_row in qst_types_df.iterrows():
                    prompt += row[1]["PROMPT_PART"] % (type_row["ID"], type_row["DESCRIPTION"]) + "\n"
            else:        
                prompt += row[1]["PROMPT_PART"] + "\n"
            
            if self.prompt_version != "2.0" and row[1]["TASK"] != "CONVERT":
                break

           
        if len(params) > 0:
            prompt = prompt % tuple(params)

        return prompt
