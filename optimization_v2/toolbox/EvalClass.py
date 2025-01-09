import os 
import json
import math
import torch
from typing import Dict, List, Literal, Optional, Tuple, Union
from optimization_v2.toolbox.SearchSpace import SearchSpace, Solution
from datetime import datetime


experiments = {
    "experiment1": {
        "name" : "experiment1",
        "lora_r" : 8,
        "lora_alpha" : 16,
        "dropout" : 0.05,
        "min_lr" : 1e-5
    },
    "experiment2": {
        "name" : "experiment2",
        "lora_r" : 4,
        "lora_alpha" : 10,
        "dropout" : 0.1,
        "min_lr" : 1e-4
    }
}

class ModelEval:
    def __init__(self,
                search_space : SearchSpace = SearchSpace(mode="base"), 
                model_id : str ="meta-llama/Llama-3.2-1B",
                experiment_name : str = "experiment1",
                dev_run : str = "fake"):
        self.model_id = model_id
        self.tasks : List = ["hellaswag","mmlu"]
        self.space = search_space
        self.folder = experiment_name
        self.result_folder = f"eval_{self.folder}"
        self.dev_run = dev_run
        self.epochs = 1

    def clean_and_add_score(self,
                            results_folder,
                            x : Solution):
        # get results and clean it
        with open(f"{results_folder}/results.json", "r") as f:
            evaluation = json.load(f)
        res = evaluation["results"]
        cleaned_results = {}
        for task in self.tasks:
            cleaned_results[task] = res[task]
        return cleaned_results
        # add score to save

        


    def evaluate(self,
                 folder : str = "meta-llama/Llama-3.2-1B") -> dict:
        # evaluation string
        tasks_str = "'"
        for task in self.tasks:
            tasks_str += task + ","
        tasks_str = tasks_str[:-1] + "'"
        if folder == self.model_id:
            self.result_folder = "evaluation"

        eval_string = (f"litgpt evaluate "+ # command
                    f"{folder}/final --out_dir {self.result_folder} "+ # path management
                    f"--tasks  {tasks_str} " #tasks definition
                    )
        os.system(eval_string)

        results = self.clean_and_add_score(self.result_folder)

        return results

    def train(self,
              lora_r, lora_alpha, dropout, min_lr, weight_decay):
        
        # training string
        optimizer_config = ("'{'class_path': 'torch.optim.AdamW', 'init_args': {"+
                f"'lr': {min_lr}, 'weight_decay': {weight_decay}, 'betas': [{0.9}, {0.999}]"+
                "}} '")
        training_string = (f"litgpt finetune "+ #command
                           f"{self.model_id} --out_dir {self.folder}"+ #path management
                           f" --devices {torch.cuda.device_count()}   --precision bf16-true "+ #global parameter of the training
                           f"--train.epochs {self.epochs} --train.lr_warmup_steps 100 --optimizer {optimizer_config} "+ #Training args
                           f"--eval.interval 1000 "+#Eval args
                           f"--lora_key true --lora_value true --lora_query true --lora_head true "+#lora application
                           f"--lora_r {lora_r} --lora_alpha {lora_alpha} --lora_dropout {dropout} " #hyperparameter
                           )
        
        os.system(training_string)
        
    def cleaning(self):

        cleaning_string = f"rm -rf {self.folder} {self.result_folder} eval"
        os.system(cleaning_string)
        os.system("rm -rf eval")
        
    def run(self,
            x : Solution):
        lora_r, lora_alpha, dropout, min_lr, weight_decay = x.get_values()

        x.opening_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.train(lora_r, lora_alpha, dropout, min_lr, weight_decay)
        x.end_training_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        results = self.evaluate(self.folder)
        x.ending_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')


        x.add_score(results)
        x.save()

        # return acc (normalized or not) for hpo
        loop_results = results[self.tasks[0]]
        try : #return acc_norm if available
            return loop_results["acc_norm,none"]
        except KeyError:
            return loop_results["acc,none"]

    def train_and_evaluate(self,
                           x : Solution) -> float:
        
        # fake run for testing
        if self.dev_run == "fake": 
            print("Running fake function")
            return x.speed_run()
        
        # get values from solution
        lora_r, lora_alpha, dropout, min_lr, weight_decay = x.get_values()

        # training string
        optimizer_config = ("'{'class_path': 'torch.optim.AdamW', 'init_args': {"+
                f"'lr': {min_lr}, 'weight_decay': {weight_decay}, 'betas': [{0.9}, {0.999}]"+
                "}} '")
        training_string = (f"litgpt finetune "+ #command
                           f"{self.model_id} --out_dir {self.folder}"+ #path management
                           f" --devices {torch.cuda.device_count()}   --precision bf16-true "+ #global parameter of the training
                           f"--train.epochs {self.epochs} --train.lr_warmup_steps 100 --optimizer {optimizer_config} "+ #Training args
                           f"--eval.interval 1000 "+#Eval args
                           f"--lora_key true --lora_value true --lora_query true --lora_head true "+#lora application
                           f"--lora_r {lora_r} --lora_alpha {lora_alpha} --lora_dropout {dropout} " #hyperparameter
                           )
        
        # evaluation string
        tasks_str = "'"
        for task in self.tasks:
            tasks_str += task + ","
        tasks_str = tasks_str[:-1] + "'"
        eval_string = (f"litgpt evaluate "+ # command
                       f"{self.folder}/final --out_dir eval_{self.folder} "+ # path management
                       f"--tasks  {tasks_str} " #tasks definition
                       )
        
        # run and timestamp
        x.opening_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        os.system(training_string)
        x.end_training_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        os.system(eval_string)
        x.ending_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # get results and clean it
        with open(f"eval_{self.folder}/results.json", "r") as f:
            evaluation = json.load(f)
        res = evaluation["results"]
        cleaned_results = {}
        for task in self.tasks:
            cleaned_results[task] = res[task]

        # add score to save
        x.add_score(cleaned_results)
        x.save()

        # cleaning
        cleaning_string = f"rm -rf {self.folder} eval_{self.folder}"
        os.system(cleaning_string)
        os.system("rm -rf eval")

        # return acc (normalized or not) for hpo
        loop_results = cleaned_results[self.tasks[0]]
        try : #return acc_norm if available
            return loop_results["acc_norm,none"]
        except KeyError:
            return loop_results["acc,none"]