import pandas as pd
import os
import sys
sys.path.append('\\'.join(os.getcwd().split('\\')[:-1])+'\\src')
from src.models.ModelEvaluator import ModelEvaluator


PROJECT_ROOT = os.getcwd()

def main():

    print("=================================================")
    print("                 MODEL EVALUATION                ")
    print("=================================================")

    # TODO:
    # 1. For each folder in the models directory
    #   4. Compute ROUGE scores -> TO BE IMPLEMENTED
    models_path = os.path.join(PROJECT_ROOT, "models")
    results_path = os.path.join(PROJECT_ROOT, "results")

    for subfolder in os.listdir(models_path):
        experiment_path = os.path.join(models_path, subfolder)

        if os.path.isdir(experiment_path):
            evaulator = ModelEvaluator()
            evaulator.load_data(project_root=PROJECT_ROOT, experiment_id=subfolder)
            
            exp_results_path = os.path.join(results_path, subfolder)
            
            if not os.path.exists(exp_results_path):
                os.makedirs(exp_results_path)
            
            print(f"Experiment ID: {subfolder}")
            print("\t - Computing statistics...")
            evaulator.compute_statistics(project_root=PROJECT_ROOT, results_dir=exp_results_path)

            print("\t - Computing BLEU scores...")
            evaulator.compute_bleu_scores(project_root=PROJECT_ROOT, results_dir=exp_results_path)

            print("\t - Computing BLEU scores distribution...")
            evaulator.compute_bleu_score_distribution(results_dir=exp_results_path)

            print("\t - Computing ROUGE scores...")
            evaulator.compute_rouge_scores(project_root=PROJECT_ROOT, results_dir=exp_results_path)

            

    print("=================================================")
    print("                 END OF EVALUATION               ")
    print("=================================================")
    

if __name__ == '__main__':
    main()
