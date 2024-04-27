import pandas as pd
import os




def main():

    print("Start processing data...")

    # ------------
    # Read the Questionnaires and corresponding Questions 
    # ------------
    project_root = os.getcwd()
    questionnaire_path = os.path.join(project_root, "data", "raw", "TF_QST_QUESTIONNAIRES.csv")
    question_path = os.path.join(project_root, "data", "raw", "TF_QST_QUESTIONS.csv")
    answers_path = os.path.join(project_root, "data", "raw", "TF_QST_ANSWERS.csv")
    question_types_path = os.path.join(project_root, "data", "raw", "TF_QST_QUESTION_TYPES.csv")

    questionnaires = pd.read_csv(questionnaire_path, encoding='latin1')
    questions = pd.read_csv(question_path, encoding='latin1')
    answers = pd.read_csv(answers_path, encoding='latin1')
    question_types = pd.read_csv(question_types_path, encoding='latin1')

    print("     1. Data loaded successfully.")

    # ------------
    # Compute the global questions' type for each questionnaire
    # ------------
    questionnaires["QUESTION_TYPE"] = 0
    for idx, row in questionnaires.iterrows():
        questionnaire_id = row["ID"]
        questionnaire_questions = questions[questions["QUESTIONNAIRE_ID"] == questionnaire_id]
        question_types_list = questionnaire_questions["TYPE_ID"].unique()
        

        if (question_types_list.size == 1):
            questionnaires.at[idx, "QUESTION_TYPE"] = question_types_list[0]

    print("     2. Global question type added successfully.")

    # ------------
    # Save dataset in the processed folder
    # ------------
    questionnaires.to_csv(os.path.join(project_root, "data", "processed", "TF_QST_QUESTIONNAIRES.csv"), index=False)
    questions.to_csv(os.path.join(project_root, "data", "processed", "TF_QST_QUESTIONS.csv"), index=False)
    answers.to_csv(os.path.join(project_root, "data", "processed", "TF_QST_ANSWERS.csv"), index=False)
    question_types.to_csv(os.path.join(project_root, "data", "processed", "TF_QST_QUESTION_TYPES.csv"), index=False)
    
    print("     3. Saved in the 'data/processed' folder.")

    print("End processing data.")


if __name__ == '__main__':
    main()
