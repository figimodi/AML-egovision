from datetime import datetime
from statistics import mean
from utils.logger import logger
import torch.nn.parallel
import torch.optim
import torch
from utils.loaders import ActionSenseDataset
from utils.args_emg import args
from utils.utils import pformat_dict
from models import FinalClassifierEMG, LSTM, LeNet5
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
    specto_model = {'specto': LeNet5()}

    hyperparameters = {
        'EMG': DotDict({'lr': 0.2, 'sgd_momentum': 0.9, 'weight_decay': 0.0001}),
        'RGB': DotDict({'lr': 0.3, 'sgd_momentum': 0.95, 'weight_decay': 0.0002}),
        'specto': DotDict({'lr': 0.3, 'sgd_momentum': 0.95, 'weight_decay': 0.0002}),
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
    
    specto_classifier = tasks.ActionRecognition("action-classifier", specto_model, args.batch_size,
                                                args.total_batch, args.models_dir, num_classes,
                                                1, hyperparameters, args=args )
    specto_classifier.load_on_gpu(device)
    specto_classifier.load_last_model('./saved_models/specto_pretrained')

    val_loader = torch.utils.data.DataLoader(
        ActionSenseDataset('test', ['EMG', 'RGB', 'specto'], 'action-net/rgb-video', args.emg_path),
        batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=False
    )

    save_logits(EMG_classifier, RGB_classifier, specto_classifier, val_loader, device)

def save_logits(EMG_classifier, RGB_classifier, specto_classifier, val_loader, device):
    EMG_classifier.reset_acc()
    EMG_classifier.train(False)
    RGB_classifier.reset_acc()
    RGB_classifier.train(False)
    specto_classifier.reset_acc()
    specto_classifier.train(False)
    
    results = []

    with torch.no_grad():
        for i_val, (data, label) in enumerate(val_loader):
            label = label.to(device)

            clips = {}
            clips['EMG'] = data['EMG'].to(device).double()
            clips['RGB'] = data['RGB'].to(device).double()
            clips['specto'] = data['specto'].to(device).double()
            emg_output, _ = EMG_classifier(clips)
            rgb_output, _ = RGB_classifier(clips)
            specto_output, _ = specto_classifier(clips)
            
            # Store the results
            results.append({
                'label': label.cpu().numpy(),
                'emg_logits': emg_output['EMG'].cpu().numpy(),
                'rgb_logits': rgb_output['RGB'].cpu().numpy(),
                'specto_logits': specto_output['specto'].cpu().numpy(),
            })

    save_dir = './saved_logits'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'logits.pkl')
    
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
   
    return 

def compute_accuracy(alpha, beta, gamma):
    data = pd.DataFrame(pd.read_pickle('saved_logits/logits.pkl'))

    labels = []
    top1_predictions = []
    top5_predictions = []

    for i, row in data.iterrows():
        true_labels = row['label']
        emg_logits = row['emg_logits']
        rgb_logits = row['rgb_logits']
        specto_logits = row['specto_logits']
        
        logits = alpha * emg_logits + beta * rgb_logits + gamma * specto_logits
        top1_index = np.argmax(logits, axis=1)
        top5_indices = np.argsort(logits, axis=1)[:, -5:]

        labels.append(true_labels)
        top1_predictions.append(top1_index)
        top5_predictions.append(np.array([true_labels[i] in top5_indices[i] for i in range(logits.shape[0])]))

    top1_predictions = np.concatenate(top1_predictions)
    top5_predictions = np.concatenate(top5_predictions)
    labels = np.concatenate(labels)

    accuracy_top1_predictions = np.array(top1_predictions == labels).mean()
    accuracy_top5_predictions = top5_predictions.mean()

    return accuracy_top1_predictions, accuracy_top5_predictions

def find_best_alpha_beta_gamma():
    # Define the range of alpha values (beta will be 1 - alpha)
    alpha_values = np.linspace(0, 1, 20)
    best_alpha = None
    best_beta = None
    best_gamma = None
    best_top1_accuracy = 0
    best_top5_accuracy = 0

    # EMG + RGB
    for alpha in alpha_values:
        beta = 1 - alpha
        gamma = 0
        top1_acc, top5_acc = compute_accuracy(alpha, beta, gamma)

        if top1_acc > best_top1_accuracy:
            best_top1_accuracy = top1_acc
            best_top5_accuracy = top5_acc
            best_alpha = alpha
            best_beta = beta

    print(f'EMG+RGB')
    print(f'Best alpha: {best_alpha:.2f}, Best beta: {best_beta:.2f}')
    print(f'Best Top-1 accuracy: {best_top1_accuracy*100:.3f}')
    print(f'Best Top-5 accuracy: {best_top5_accuracy*100:.3f}')

    # RGB + specto
    best_alpha = None
    best_beta = None
    best_gamma = None
    best_top1_accuracy = 0
    best_top5_accuracy = 0
    for beta in alpha_values:
        alpha = 0
        gamma = 1 - beta
        top1_acc, top5_acc = compute_accuracy(alpha, beta, gamma)

        if top1_acc > best_top1_accuracy:
            best_top1_accuracy = top1_acc
            best_top5_accuracy = top5_acc
            best_beta = beta
            best_gamma = gamma

    print(f'RGB+specto')
    print(f'Best beta: {best_beta:.2f}, Best gamma: {best_gamma:.2f}')
    print(f'Best Top-1 accuracy: {best_top1_accuracy*100:.3f}')
    print(f'Best Top-5 accuracy: {best_top5_accuracy*100:.3f}')

    # EMG + specto
    best_alpha = None
    best_beta = None
    best_gamma = None
    best_top1_accuracy = 0
    best_top5_accuracy = 0
    for alpha in alpha_values:
        gamma = 1 - alpha
        beta = 0
        top1_acc, top5_acc = compute_accuracy(alpha, beta, gamma)

        if top1_acc > best_top1_accuracy:
            best_top1_accuracy = top1_acc
            best_top5_accuracy = top5_acc
            best_alpha = alpha
            best_gamma = gamma

    print(f'EMG+specto')
    print(f'Best alpha: {best_alpha:.2f}, Best gamma: {best_gamma:.2f}')
    print(f'Best Top-1 accuracy: {best_top1_accuracy*100:.3f}')
    print(f'Best Top-5 accuracy: {best_top5_accuracy*100:.3f}')

    # EMG + RGB + specto
    best_alpha = None
    best_beta = None
    best_gamma = None
    best_top1_accuracy = 0
    best_top5_accuracy = 0
    for alpha in alpha_values:
        beta_values = np.linspace(alpha, 1, 20)
        for beta in beta_values:
            gamma = 1 - beta
            top1_acc, top5_acc = compute_accuracy(alpha, beta, gamma)

            if top1_acc > best_top1_accuracy:
                best_top1_accuracy = top1_acc
                best_top5_accuracy = top5_acc
                best_alpha = alpha
                best_beta = beta
                best_gamma = gamma

    print(f'EMG+RGB+specto')
    print(f'Best alpha: {best_alpha:.2f}, Best beta: {best_beta:.2f}, Best gamma: {best_gamma:.2f}')
    print(f'Best Top-1 accuracy: {best_top1_accuracy*100:.3f}')
    print(f'Best Top-5 accuracy: {best_top5_accuracy*100:.3f}')

if __name__ == '__main__':
    main()
    # find_best_alpha_beta_gamma()
    acc1, acc5 = compute_accuracy(1, 0, 0)
    print(acc1)
    print(acc5)
