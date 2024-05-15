import os
import sys
sys.path.append('\\'.join(os.getcwd().split('\\')[:-1])+'\\src')
from src.models.ModelEvaluator import ModelEvaluator


PROJECT_ROOT = os.getcwd()

def main():

    print("=================================================")
    print("                 MODEL EVALUATION                ")
    print("=================================================")

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
            print("\t 1. Computing statistics...")
            evaulator.compute_statistics(project_root=PROJECT_ROOT, results_dir=exp_results_path)

            print("\t 2. Computing BLEU scores...")
            evaulator.compute_bleu_scores(project_root=PROJECT_ROOT, results_dir=exp_results_path)

            print("\t 3. Computing BLEU scores distribution...")
            evaulator.compute_bleu_score_distribution(results_dir=exp_results_path)

            print("\t 4. Computing ROUGE scores...")
            evaulator.compute_rouge_scores(project_root=PROJECT_ROOT, results_dir=exp_results_path)

            print("\t 5. Computing ROUGE-L (F1) scores distribution...")
            evaulator.compute_rouge_score_distribution(results_dir=exp_results_path)

            print("\t 6. Computing Syntactic Similarity...")
            evaulator.compute_syntactic_similarities(project_root=PROJECT_ROOT, results_dir=exp_results_path)

    print("=================================================")
    print("                 END OF EVALUATION               ")
    print("=================================================")
    

if __name__ == '__main__':
    main()
