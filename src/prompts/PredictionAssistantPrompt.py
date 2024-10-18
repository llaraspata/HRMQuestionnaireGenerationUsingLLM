import os
import pandas as pd
import numpy as np

class PredictionAssistantPrompt:
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
        
        prompts_df = prompts_df[prompts_df["TYPE"] == 'A']
        self.prompts_df = prompts_df[prompts_df["VERSION"] == np.float64(self.prompt_version)][self.ESSENTIAL_COLUMNS].sort_values(by="ORDER")

    
    def build_prompt(self, params=[], prompt_task=""):
        prompt = ""

        if self.prompts_df is None or len(self.prompts_df) == 0:
            prompt = "%s"
        else:
            for row in self.prompts_df.iterrows():       
                if self.prompt_version == "2.0" and row[1]["TASK"] != prompt_task:
                    continue

                prompt += row[1]["PROMPT_PART"] + "\n"

                if self.prompt_version == "2.0" and row[1]["TASK"] != "CONVERT":
                    break

        if len(params) > 0:
            prompt = prompt % tuple(params)

        return prompt
