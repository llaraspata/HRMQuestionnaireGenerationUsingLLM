import os
import shutil
import sys
sys.path.append('\\'.join(os.getcwd().split('\\')[:-1])+'\\src')
from src.evaluation.QuestionnairesEvaluator import QuestionnairesEvaluator
from src.evaluation.ModelEvaluator import ModelEvaluator


PROJECT_ROOT = os.getcwd()

def main():

    models = ["LLaMa", "Mistral"]
    tasks = ["Survey"]
    prompt_versions = ["1.0"]

    for task in tasks:
        for prompt_version in prompt_versions:
            for model in models:
                if model == "GPT":
                    continue
                evaluate(task, prompt_version, model)


def evaluate(task, prompt_version, model):

    print("=================================================")
    print(f"                 {model} EVALUATION              ")
    print("=================================================")

    models_path = os.path.join(PROJECT_ROOT, "models", task, prompt_version, model)
    results_path = os.path.join(PROJECT_ROOT, "results", task, prompt_version, model)

    model_evaluator = ModelEvaluator(results_dir=results_path)

    for subfolder in os.listdir(models_path):
        experiment_path = os.path.join(models_path, subfolder)
        
        if os.path.isdir(experiment_path):
            questionnaires_evaluator = QuestionnairesEvaluator()
            questionnaires_evaluator.load_data(project_root=PROJECT_ROOT, task=task, prompt_version=prompt_version, model=model, experiment_id=subfolder)
            
            exp_results_path = os.path.join(results_path, subfolder)

            if not os.path.exists(exp_results_path):
                os.makedirs(exp_results_path)
            else:
                clean_folder(exp_results_path)
            
            print(f"Experiment ID: {subfolder}")
            print("\t 1. Computing statistics...")
            questionnaires_evaluator.compute_statistics(project_root=PROJECT_ROOT, results_dir=exp_results_path)

            print("\t 2. Computing BLEU scores...")
            questionnaires_evaluator.compute_bleu_scores(project_root=PROJECT_ROOT, results_dir=exp_results_path)

            print("\t 3. Computing BLEU scores distribution...")
            questionnaires_evaluator.compute_bleu_score_distribution(results_dir=exp_results_path)

            print("\t 4. Computing ROUGE scores...")
            questionnaires_evaluator.compute_rouge_scores(project_root=PROJECT_ROOT, results_dir=exp_results_path)

            print("\t 5. Computing ROUGE-L (F1) scores distribution...")
            questionnaires_evaluator.compute_rouge_score_distribution(results_dir=exp_results_path)

            print("\t 6. Computing Syntactic Similarity...")
            questionnaires_evaluator.compute_syntactic_similarities(project_root=PROJECT_ROOT, results_dir=exp_results_path)

            print("\t 7. Computing Semantic Similarity...")
            questionnaires_evaluator.compute_semantic_similarity(project_root=PROJECT_ROOT, results_dir=exp_results_path)

            print("\t 8. Computing Serendipity...")
            questionnaires_evaluator.compute_serendipity(project_root=PROJECT_ROOT, results_dir=exp_results_path)

            print("\t 9. Computing Question Type Variability...")
            questionnaires_evaluator.compute_qst_type_variability(project_root=PROJECT_ROOT, results_dir=exp_results_path)

    print("-----------------------------------")
    print("Globally evaluating the model...")
    model_evaluator.evaluate()
    model_evaluator.compute_time_token_cost(project_root=PROJECT_ROOT, models_path=models_path, results_dir=results_path, 
                                            task=task, prompt_version=prompt_version, model=model)
    
    print("=================================================")
    print("                 END OF EVALUATION               ")
    print("=================================================")


def clean_folder(folder_path):
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(e)


if __name__ == '__main__':
    main()
