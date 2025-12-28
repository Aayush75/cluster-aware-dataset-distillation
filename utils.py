import json
import os
import argparse
import torch.nn as nn
import torch

def save_arguments(script_name, args):
    directory = "../log/"
    filename = os.path.join(directory, f"{args.exp_name}.json")
    os.makedirs(directory, exist_ok=True)

    # If the JSON file exists, load its data. If not, initialize an empty list.
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
    else:
        data = []

    found = False
    for entry in data:
        # Skip entries that don't have a "script" key (e.g., accuracy logs)
        if not isinstance(entry, dict) or "script" not in entry:
            continue
        if entry["script"] == script_name:
            # Update the arguments for this script
            entry["args"] = vars(args)
            found = True
            break

    # If script_name was not found in the list, append a new entry
    if not found:
        entry = {
            "script": script_name,
            "args": vars(args)
        }
        data.append(entry)

    # Save the updated data back to the JSON file
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


DATASETS = {"imagenet", "imagenet_data", "tiny-imagenet", "imagenette", "cifar100", "wolf", "birds", "fruits", "cats", "a", "b", "c", "d", "e"}


def get_num_class_map(datasets):
    num_class_map = dict()
    for d in datasets:
        if d == "imagenet" or d == "imagenet_data":
            num_class_map[d] = 1000
        elif d == "tiny-imagenet":
            num_class_map[d] = 200
        elif d == "cifar100":
            num_class_map[d] = 100
        else:
            num_class_map[d] = 10
    return num_class_map

# Example usage:
# model = torchvision.models.resnet50()
# replace_batchnorm(model, num_classes=1000)