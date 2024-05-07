from datetime import datetime
from statistics import mean
from utils.logger import logger
import torch.nn.parallel
import torch.optim
import torch
from utils.loaders import ActionSenseDataset
from utils.args_emg import args
from utils.utils import pformat_dict
import utils
import numpy as np
import os
import models as model_list
import tasks
import wandb

# global variables among training functions

training_iterations = 0
args_mod = None
modalities=[args.modality]

np.random.seed(13696641)
torch.manual_seed(13696641)

torch.set_default_dtype(torch.float64)

def init_operations():
    """
    parse all the arguments, generate the logger, check gpus to be used and wandb
    """
    # logger.info("Running with parameters: " + pformat_dict(args, indent=1))

    # this is needed for multi-GPUs systems where you just want to use a predefined set of GPUs
    if args.gpus is not None:
        logger.debug('Using only these GPUs: {}'.format(args.gpus))
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus)

    # wanbd logging configuration
    if args.wandb_name is not None:
        wandb.init(group=args.wandb_name, dir=args.wandb_dir)
        wandb.run.name = args.name + "_" + args.shift.split("-")[0] + "_" + args.shift.split("-")[-1]

def main():
    global training_iterations, modalities, args_mod
    init_operations()
    
    modality = args.modality
    args_mod = args.modalities[modality]
    num_classes = 20
    
    # device where everything is run
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # these dictionaries are for more multi-modal training/testing, each key is a modality used
    models = {}
    
    print(model_list)
    
    logger.info("Instantiating models per modality")
    for m in modalities:
        logger.info('Model: {}\tModality: {}'.format(args.modalities[m].models[m].name, m))
        # notice that here, the first parameter passed is the input dimension
        # In our case it represents the feature dimensionality which is equivalent to 1024 for I3D
        models[m] = getattr(model_list, args.modalities[m].models[m].name)()

    # the models are wrapped into the ActionRecognition task which manages all the training steps
    action_classifier = tasks.ActionRecognition("action-classifier", models, args.batch_size,
                                                args.total_batch, args.models_dir, num_classes,
                                                args_mod.train.num_clips if modality!="EMG" else 1, args_mod.models, args=args)
    action_classifier.load_on_gpu(device)

    if args.action == "train":
        # resume_from argument is adopted in case of restoring from a checkpoint
        if args.resume_from is not None:
            action_classifier.load_last_model(args.resume_from)
        # define number of iterations I'll do with the actual batch: we do not reason with epochs but with iterations
        # i.e. number of batches passed
        # notice, here it is multiplied by tot_batch/batch_size since gradient accumulation technique is adopted
        training_iterations = args_mod.train.num_iter * (args.total_batch // args.batch_size)
        # all dataloaders are generated here
        
        print("Train loader")
        
        train_loader = torch.utils.data.DataLoader(
            ActionSenseDataset('train', [args.modality]),
            batch_size=args.batch_size, shuffle=True, num_workers=args_mod.dataset.workers, pin_memory=True, drop_last=True
        )
        
        print("Val loader")

        val_loader = torch.utils.data.DataLoader(
                ActionSenseDataset('test', [args.modality]),
            batch_size=args.batch_size, shuffle=False, num_workers=args_mod.dataset.workers, pin_memory=True, drop_last=False
        )
        
        train(action_classifier, train_loader, val_loader, device, num_classes)

    elif args.action == "validate":
        if args.resume_from is not None:
            action_classifier.load_last_model(args.resume_from)
        
        val_loader = torch.utils.data.DataLoader(
            ActionSenseDataset("test", [args.modality]),
            batch_size=args.batch_size, shuffle=False, num_workers=args_mod.dataset.workers, pin_memory=True, drop_last=False
        )

        validate(action_classifier, val_loader, device, action_classifier.current_iter, num_classes)


def train(action_classifier, train_loader, val_loader, device, num_classes):
    """
    function to train the model on the test set
    action_classifier: Task containing the model to be trained
    train_loader: dataloader containing the training data
    val_loader: dataloader containing the validation data
    device: device on which you want to test
    num_classes: int, number of classes in the classification problem
    """
    
    
    global training_iterations, modalities, args_mod

    data_loader_source = iter(train_loader)
    action_classifier.train(True)
    action_classifier.zero_grad()
    iteration = action_classifier.current_iter * (args.total_batch // args.batch_size)

    # the batch size should be total_batch but batch accumulation is done with batch size = batch_size.
    # real_iter is the number of iterations if the batch size was really total_batch
    for i in range(iteration, training_iterations):
        # iteration w.r.t. the paper (w.r.t the bs to simulate).... i is the iteration with the actual bs( < tot_bs)
        real_iter = (i + 1) / (args.total_batch // args.batch_size)
        if real_iter == args_mod.train.lr_steps:
            # learning rate decay at iteration = lr_steps
            action_classifier.reduce_learning_rate()
        # gradient_accumulation_step is a bool used to understand if we accumulated at least total_batch
        # samples' gradient
        gradient_accumulation_step = real_iter.is_integer()

        """
        Retrieve the data from the loaders
        """
        # start_t = datetime.now()
        # the following code is necessary as we do not reason in epochs so as soon as the dataloader is finished we need
        # to redefine the iterator
        
        try:
            source_data, source_label = next(data_loader_source)
        except StopIteration:
            data_loader_source = iter(train_loader)
            source_data, source_label = next(data_loader_source)
        
        # end_t = datetime.now()
        # logger.info(f"Iteration {i}/{training_iterations} batch retrieved! Elapsed time = "
        #             f"{(end_t - start_t).total_seconds() // 60} m {(end_t - start_t).total_seconds() % 60} s")

        ''' Action recognition'''
        source_label = source_label.to(device)
        data = {}

        if args.modality == 'RGB':
            if args.models.RGB.model == 'LSTM' or args.models.RGB.model == 'RNN':
                # skip aggregation but concatenate features
                # source_data['RGB'].shape = (32, 1, 5120) containing the 5x1024 clips flattened
                source_data['RGB'] = source_data['RGB'].view(32, -1).unsqueeze(1)
            else:
                # aggregate features along temporal axis with a pooling layer
                # pooling_layer = torch.nn.MaxPool2d(kernel_size=(5, 1))
                # source_data['RGB'].shape = (32, 1, 1024)
                pooling_layer = torch.nn.AvgPool2d(kernel_size=(5, 1))
                source_data['RGB'] = pooling_layer(source_data['RGB'])

                # aggregate features along the temporal axis with a convolutional layer
                # source_data['RGB'].shape = (32, 1, 1024)
                # conv_layer = torch.nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=5)
                # data_permuted = source_data['RGB'].permute(0, 2, 1)
                # conv_output = conv_layer(data_permuted)
                # conv_output_permuted = conv_output.permute(0, 2, 1)
                # source_data['RGB'] = conv_output_permuted+
            
        for m in modalities:
            if m == 'EMG':
                data[m] = source_data[m].to(device)
            
        logits, _ = action_classifier.forward(data)
        action_classifier.compute_loss(logits, source_label, loss_weight=1)
        action_classifier.backward(retain_graph=False)
        action_classifier.compute_accuracy(logits, source_label)

        if gradient_accumulation_step:
            logger.info("[%d/%d]\tlast Verb loss: %.4f\tMean verb loss: %.4f\tAcc@1: %.2f%%\tAccMean@1: %.2f%%" %
                        (real_iter, args_mod.train.num_iter, action_classifier.loss.val, action_classifier.loss.avg,
                         action_classifier.accuracy.val[1], action_classifier.accuracy.avg[1]))

            action_classifier.check_grad()
            action_classifier.step()
            action_classifier.zero_grad()

        if gradient_accumulation_step and real_iter % args_mod.train.eval_freq == 0:
            val_metrics = validate(action_classifier, val_loader, device, int(real_iter), num_classes)

            if val_metrics['top1'] <= action_classifier.best_iter_score:
                logger.info("New best accuracy {:.2f}%"
                            .format(action_classifier.best_iter_score))
            else:
                logger.info("New best accuracy {:.2f}%".format(val_metrics['top1']))
                action_classifier.best_iter = real_iter
                action_classifier.best_iter_score = val_metrics['top1']

            action_classifier.save_model(real_iter, val_metrics['top1'], prefix=None)
            action_classifier.train(True)


def validate(model, val_loader, device, it, num_classes):
    """
    function to validate the model on the test set
    model: Task containing the model to be tested
    val_loader: dataloader containing the validation data
    device: device on which you want to test
    it: int, iteration among the training num_iter at which the model is tested
    num_classes: int, number of classes in the classification problem
    """
    global modalities

    model.reset_acc()
    model.train(False)
    logits = {}

    # Iterate over the models
    with torch.no_grad():
        for i_val, (data, label) in enumerate(val_loader):
            label = label.to(device)

            if modalities[0]=='EMG' and len(modalities)==1:
                clips = {}
                for m in modalities:
                    clips[m] = data[m].to(device)
                    output, _ = model(clips)
                    logits[m] = output[m]
            else:
                for m in modalities:
                    batch = data[m].shape[0]
                    logits[m] = torch.zeros((args.test.num_clips, batch, num_classes)).to(device)

                clip = {}
                for i_c in range(args.test.num_clips):
                    for m in modalities:
                        clip[m] = data[m][:, i_c].to(device)

                    output, _ = model(clip)
                    for m in modalities:
                        logits[m][i_c] = output[m]

                for m in modalities:
                    logits[m] = torch.mean(logits[m], dim=0)

            model.compute_accuracy(logits, label)

            if (i_val + 1) % (len(val_loader) // 5) == 0:
                logger.info("[{}/{}] top1= {:.3f}% top5 = {:.3f}%".format(i_val + 1, len(val_loader), model.accuracy.avg[1], model.accuracy.avg[5]))

        class_accuracies = [(x / y) * 100 for x, y in zip(model.accuracy.correct, model.accuracy.total) if y>0]
        logger.info('Final accuracy: top1 = %.2f%%\ttop5 = %.2f%%' % (model.accuracy.avg[1], model.accuracy.avg[5]))
        for i_class, class_acc in enumerate(class_accuracies):
            logger.info('Class %d = [%d/%d] = %.2f%%' % (i_class, int(model.accuracy.correct[i_class]), int(model.accuracy.total[i_class]), class_acc))

    logger.info('Accuracy by averaging class accuracies (same weight for each class): {}%'
                .format(np.array(class_accuracies).mean(axis=0)))
    test_results = {'top1': model.accuracy.avg[1], 'top5': model.accuracy.avg[5],
                    'class_accuracies': np.array(class_accuracies)}

    with open(os.path.join(args.log_dir, f'val_precision_{modalities[0]}.txt'), 'a+') as f:
        f.write("[%d/%d]\tAcc@top1: %.2f%%\n" % (it, args.train.num_iter, test_results['top1']))

    return test_results


if __name__ == '__main__':
    main()
