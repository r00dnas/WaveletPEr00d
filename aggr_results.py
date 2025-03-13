import os
import numpy as np 
import json 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type = str)
parser.add_argument("--model_type", type = str)

args = parser.parse_args()


file_path = os.path.join(f"logs/ogbg-mol{args.dataset_name}", f"{args.model_type}_results.json")
with open(file_path, "r") as file_:
    data = json.load(file_)

items = list(data.items())

results = []
for key, value in items:
    if "idx" in value.keys():
        results.append((value['idx'], value['test_score']))


results = sorted(results, key=lambda x: x[1])[::-1]

top_5_results = list(map(lambda x: x[1], results[:3]))

print("Dataset Name: ", args.dataset_name)
print("Model: ", args.model_type)
print(f"Mean: {np.mean(top_5_results)} | Std: {np.std(top_5_results)}")
print()
