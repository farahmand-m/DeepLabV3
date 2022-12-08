import copy
import csv
import os
import time

import evaluate
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


metric = evaluate.load("mean_iou")

# You should redefine the following. I'm not going to bother with reading it from a file or something.
id2label = {0: 'Background', 1: 'Foreground'}


# the following implementation is borrowed from HuggingFace's tutorial on finetuning SegFormer.
def compute_metrics(predictions, targets):
    # currently using _compute instead of compute
    # see this issue for more info: https://github.com/huggingface/evaluate/pull/328#issuecomment-1286866576
    metrics = metric._compute(
            predictions=predictions,
            references=targets,
            num_labels=len(id2label),
            ignore_index=0,
            reduce_labels=False,
        )

    # add per category metrics as individual key-value pairs
    per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
    per_category_iou = metrics.pop("per_category_iou").tolist()

    metrics.update({f"accuracy_{id2label[i]}": v for i, v in enumerate(per_category_accuracy)})
    metrics.update({f"iou_{id2label[i]}": v for i, v in enumerate(per_category_iou)})

    return metrics


def train_model(model, criterion, dataloaders, optimizer, metrics, bpath,
                num_epochs):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    # Use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Initialize the log file for training and testing loss and metrics
    fieldnames = ['epoch', 'Train_loss', 'Test_loss'] + \
        [f'Train_{m}' for m in metrics.keys()] + \
        [f'Test_{m}' for m in metrics.keys()]
    with open(os.path.join(bpath, 'log.csv'), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    for epoch in range(1, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        # Each epoch has a training and validation phase
        # Initialize batch summary
        batchsummary = {a: [0] for a in fieldnames}

        predictions = []
        targets = []

        for phase in ['Train', 'Test']:
            if phase == 'Train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            # Iterate over data.
            for sample in tqdm(iter(dataloaders[phase])):
                inputs = sample['image'].to(device)
                masks = sample['mask'].to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                if len(inputs) == 1:
                    continue

                # track history if only in train
                with torch.set_grad_enabled(phase == 'Train'):
                    outputs = model(inputs)
                    loss = criterion(outputs['out'], masks)
                    y_pred = outputs['out'].data.cpu().numpy()  # Raw Logits
                    y_true = masks.data.cpu().numpy()  # Segments Labels
                    predictions.append(y_pred)
                    targets.append(y_true)
                    for name, metric in metrics.items():
                        if name == 'f1_score':
                            Y, Y_ = y_true.ravel() > 0, y_pred.ravel() > 0
                        else:
                            Y, Y_ = y_true.ravel().astype('uint8'), y_pred.ravel()
                        try:
                            value = metric(Y, Y_)
                        except ValueError:
                            value = 0
                        batchsummary[f'{phase}_{name}'].append(value)

                    # backward + optimize only if in training phase
                    if phase == 'Train':
                        loss.backward()
                        optimizer.step()
            batchsummary['epoch'] = epoch
            epoch_loss = loss
            batchsummary[f'{phase}_loss'] = epoch_loss.item()
            print('{} Loss: {:.4f}'.format(phase, loss))

        predictions = np.concatenate(predictions, axis=0)
        predictions = np.squeeze(predictions)
        predictions = (predictions > 0).astype(np.float32)
        targets = np.concatenate(targets, axis=0)
        targets = np.squeeze(targets)
        targets = (targets > 0).astype(np.float32)

        metrics_values = compute_metrics(predictions, targets)
        print(metrics_values)

        for field in fieldnames[3:]:
            batchsummary[field] = np.mean(batchsummary[field])
        print(batchsummary)
        with open(os.path.join(bpath, 'log.csv'), 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(batchsummary)
            # deep copy the model
            if phase == 'Test' and loss < best_loss:
                best_loss = loss
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Lowest Loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
