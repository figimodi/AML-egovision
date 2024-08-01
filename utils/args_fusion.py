import os
import platform
from datetime import datetime
from omegaconf import OmegaConf
import pprint

def add_paths():
    path_conf = OmegaConf.create()
    return path_conf

# Retrieve the configs path
conf_path = os.path.join(os.path.dirname(__file__), '../configs')

# Read the cli args
cli_args = OmegaConf.from_cli()

args = OmegaConf.load(os.path.join(conf_path, "train_fusion.yaml"))

path_args = add_paths()
args = OmegaConf.merge(args, path_args)

gloab_configs = ["name", "split", "action", "modality"]
cli_args_global = {key: cli_args[key] for key in gloab_configs}
cli_args_modality = {key: cli_args[key] for key in cli_args if key not in gloab_configs}
modality = cli_args_global["modality"]

# Merge cli args into config ones
args = OmegaConf.merge(args, cli_args_global)
args.modalities = {modality:OmegaConf.merge(args.modalities[modality], cli_args_global)}

# add log directories
args.experiment_dir = os.path.join(args.name, datetime.now().strftime('%b%d_%H-%M-%S'))
if args.action != "train":
    args.log_dir = os.path.join('TEST_RESULTS', args.name)
    if args.logname is None:
        args.logname = args.action + "_" + args.modalities[modality].dataset.agent + ".log"
    else:
        args.logname = args.logname + "_" + args.modalities[modality].dataset.agent + ".log"
    args.logfile = os.path.join(args.log_dir, args.logname)
else:
    args.log_dir = os.path.join('Experiment_logs', args.experiment_dir)
    args.logfile = os.path.join(args.log_dir, args.action + ".log")
os.makedirs(args.log_dir, exist_ok=True)
if args.models_dir is None:
    args.models_dir = os.path.join("saved_models", args.experiment_dir)
if args.action != "train" and args.action != 'save' and args.resume_from is None:
    args.resume_from = os.path.join(args.models_dir, args.name)
