from datetime import datetime
from statistics import mean
from utils.logger import logger
import torch.nn.parallel
import torch.optim
import torch
from utils.loaders import ActionSenseDataset
from utils.args_emg import args
from utils.utils import pformat_dict
from models import FinalClassifierEMG, LSTM
import utils
import numpy as np
import os
import tasks
import wandb
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score


class DotDict(dict):
    """Dictionary with dot notation access to attributes."""
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{name}'")
    
    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{name}'")

def init_operations(): 
    np.random.seed(13696641)
    torch.manual_seed(13696641)
    torch.set_default_dtype(torch.float64)
    
    if args.gpus is not None:
        logger.debug('Using only these GPUs: {}'.format(args.gpus))
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus)

def main():
    init_operations()
    
    num_classes = 20
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emg_model = {'EMG': FinalClassifierEMG()}
    rgb_model = {'RGB': LSTM()}

    hyperparameters = {
        'EMG': DotDict({'lr': 0.2, 'sgd_momentum': 0.9, 'weight_decay': 0.0001}),
        'RGB': DotDict({'lr': 0.3, 'sgd_momentum': 0.95, 'weight_decay': 0.0002})
    }

    EMG_classifier = tasks.ActionRecognition("action-classifier", emg_model, args.batch_size,
                                                args.total_batch, args.models_dir, num_classes,
                                                1, hyperparameters, args=args )
    EMG_classifier.load_on_gpu(device)
    EMG_classifier.load_last_model('./saved_models/EMG_pretrained')

    RGB_classifier = tasks.ActionRecognition("action-classifier", rgb_model, args.batch_size,
                                                args.total_batch, args.models_dir, num_classes,
                                                1, hyperparameters, args=args )
    RGB_classifier.load_on_gpu(device)
    RGB_classifier.load_last_model('./saved_models/RGB_pretrained')
    
    val_loader = torch.utils.data.DataLoader(
        ActionSenseDataset('test', ['EMG', 'RGB'], args.split_path, args.emg_path),
        batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=False
    )

    save_logits(EMG_classifier, RGB_classifier, val_loader, device)

def save_logits(EMG_model, RGB_model, val_loader, device):
    EMG_model.reset_acc()
    EMG_model.train(False)
    RGB_model.reset_acc()
    RGB_model.train(False)
    
    results = []

    with torch.no_grad():
        for i_val, (data, label) in enumerate(val_loader):
            label = label.to(device)

            clips = {}
            clips['EMG'] = data['EMG'].to(device).double()
            clips['RGB'] = data['RGB'].to(device).double()
            emg_output, _ = EMG_model(clips)
            rgb_output, _ = RGB_model(clips)

            concatenated_logits = torch.cat((emg_output['EMG'], rgb_output['RGB']), dim=1).cpu().numpy()
            
            # Store the results
            results.append({
                'label': label.cpu().numpy(),
                'logits': concatenated_logits
            })

    save_dir = './saved_logits'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'concatenated_logits_test.pkl')
    
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
   
    return 

def compute_accuracy():
    data = pd.DataFrame(pd.read_pickle('saved_logits/concatenated_logits_test.pkl'))

    labels = []
    top1_predictions = []
    top5_predictions = []

    for i, row in data.iterrows():
        true_labels = row['label']
        logits = row['logits'].reshape(-1, 20, 2)
        
        # mean_logits = logits.mean(axis=2)
        mean_logits = logits[:,:,1] 
        top1_index = np.argmax(mean_logits, axis=1)
        top5_indices = np.argsort(mean_logits, axis=1)[:, -5:]


        labels.append(true_labels)
        top1_predictions.append(top1_index)
        top5_predictions.append(np.array([true_labels[i] in top5_indices[i] for i in range(logits.shape[0])]))

    top1_predictions = np.concatenate(top1_predictions)
    top5_predictions = np.concatenate(top5_predictions)
    labels = np.concatenate(labels)

    accuracy_top1_predictions = np.array(top1_predictions == labels).mean()
    accuracy_top5_predictions = top5_predictions.mean()

    print(f'The overall Top-1 accuracy using late fusion is: {accuracy_top1_predictions*100:.3f}')
    print(f'The overall Top-5 accuracy using late fusion is: {accuracy_top5_predictions*100:.3f}')

if __name__ == '__main__':
    # main()
    compute_accuracy()
