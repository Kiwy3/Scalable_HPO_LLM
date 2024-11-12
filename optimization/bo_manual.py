from zellij.core.geometry import Direct, Soo
from zellij.strategies import DBA, Bayesian_optimization
from zellij.strategies.tools.tree_search import Potentially_Optimal_Rectangle, Soo_tree_search
from zellij.strategies.tools.direct_utils import Sigma2, SigmaInf
from zellij.utils.converters import IntMinmax
from zellij.core.objective import Maximizer
from zellij.core import ContinuousSearchspace, FloatVar,IntVar, ArrayVar, Loss 
#from zellij.utils.benchmarks import himmelblau
#from model_evaluation import evaluate
import torch
import math
from pathlib import Path
import json
import pandas as pd

# Bayesian function
from botorch.models import SingleTaskGP
from botorch.models.gp_regression_mixed import MixedSingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
from botorch.acquisition.analytic import LogExpectedImprovement

# custom librairies
#import model_evaluation
#from model_evaluation import training, evaluate

hp_def = { 
   "learning_rate" : {"min" : -10,"max" : -1,"type" : "exp"},
   "lora_rank" : {"min" : 2,"max" : 32,"type" : "int"},
   "grad_batches" : {"min" : 0,"max" : 16,"type" : "int"},
   "lora_alpha" : {"min" : 16,"max" : 64,"type" : "int"},
   "lora_dropout" : {"min" : 0,"max" : 0.5,"type" : "float"},
   "weight_decay" : {"min" : 0,"max" : 0.5,"type" : "float"}, 
   }

model_dict = {
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0":"tiny-llama-1.1b",
    "meta-llama/Meta-Llama-3.1-8B":"Llama-3.1-8B",

}

def convert(x,i, hyperparameters=hp_def):
    key = list(hyperparameters.keys())[i]
    type = hyperparameters[key]["type"]
    if type == "int":
        return int(x[i].item())
    elif type == "exp":
        return math.exp(x[i].item())
    elif type == "float":
        return float(x[i].item())

#from model_evaluation import evaluate
def evaluation_function(x):
    return himmelblau(x)
    # convert x into hyperparameters
    hyperparameters = {}
    for i in range(len(hp_def.keys())):
        key = list(hp_def.keys())[i]
        hyperparameters[key] = convert(x,i)

    # save hyperparameters
    HP = {"hyperparameters" : hyperparameters,
          "experiment" : experiment}   
    with open(export_file, "a") as outfile:
        json.dump(HP, outfile)
        outfile.write('\n')
    print(model_evaluation.utils.load_hyperparameters())

    training()
    result = evaluate()

    return result["mmlu"]



def himmelblau(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


if __name__ == "__main__":
    export_file = "optimization/export.json"
    experiment = {"model_id" : "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                  "nb_device" : 2,
                  "epochs" : 1,
                  "device" : "cuda",
                  "fast_run" : False,
                  "eval_limit" : 100,
                  "calls":50,
                  "file" : "analysis/analysis2.json"
                  }
    experiment["model_name"] = model_dict[experiment["model_id"]]

    # Initiate BoTorch
    lower_bounds = torch.tensor([hp_def[key]["min"] for key in hp_def.keys()])
    upper_bounds = torch.tensor([hp_def[key]["max"] for key in hp_def.keys()])
    bounds = torch.stack((lower_bounds, upper_bounds)
    )
    
    if Path(experiment["file"]).is_file():
        data = pd.read_json(experiment["file"],lines=True)
        data = data[data.results.notnull()]
        Y = data.results.apply(lambda x: [x["mmlu"]])
        Y = torch.tensor(Y,dtype=torch.double)
        X = pd.json_normalize(data["hyperparameters"])
        X = torch.tensor(X.values,dtype=torch.double)
    else:
        X = [(lower_bounds+upper_bounds)/2].to(torch.double)
        #X = torch.tensor(X,dtype=torch.double)
        Y = torch.tensor([evaluation_function(X)],dtype=torch.double)

    print("model initialized")
    for i in range(experiment.get("calls",10)):
        print("iteration ",i,":")
        # Define the model
        print("\t creating new model")
        gp = MixedSingleTaskGP(
        train_X=X,
        train_Y=Y,
        cat_dims=[-1],
        input_transform=Normalize(d=len(hp_def.keys())),
        outcome_transform=Standardize(m=1),
        )

        # Optimize the model
        print("\t optimizing model")
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        logEI = LogExpectedImprovement(model=gp, best_f=Y.max())
        candidate, acq_value = optimize_acqf(
            logEI, bounds=bounds, q=1, num_restarts=5, raw_samples=20,
        )

        # Compute the new evaluation
        print("\t computing new evaluation")
        print("\t\t new_solution : ",candidate)
        candidate_list = [candidate[0][i].item() for i in range(len(candidate[0]))]
        score = evaluation_function(candidate_list)
        X = torch.cat((X,candidate))
        Y = torch.cat((Y,
                       torch.tensor([[score]],dtype=torch.double)))
        print("\t\t new_score : ",score)
    
    print("best solution : ",X[Y.argmax()])
    print("best score : ",Y.max())
        

    





    

