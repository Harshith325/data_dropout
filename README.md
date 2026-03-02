# Progressive Data Dropout: An Embarrassingly Simple Approach to Faster Train

Link to paper : https://arxiv.org/pdf/2505.22342

### Dependencies
- Install python3 and pytorch

### Datasets Supported
The 'data.py' file has the dataloaders for CIFAR100, CIFAR10, MNIST and IMAGENET. 

### Additional datasets 
For longtail classification, the dataset is created by the imbalance_cifar.py file. 

Running the baseline (example: model, save_path, dataset and batch size can be changed accordingly): 
```
python .\main.py --model mobilenet_v2 --mode baseline --epoch 30 --save_path cifar10_results/mobilenet_v2 --dataset cifar10 --batch_size 32 --task classification 
```

Running the Difficulty-Based Progressive Dropout (DBPD):
```
python .\main.py --model mobilenet_v2 --mode train_with_revision --epoch 30 --save_path cifar10_results/mobilenet_v2 --dataset cifar10 --batch_size 32 --start_revision 29 --task classification --threshold 0.3
```

Running the Scheduled Match Random Dropout (SMRD) : 
```
python .\main.py --model mobilenet_v2 --mode train_with_random --epoch 30 --save_path cifar10_results/mobilenet_v2 --dataset cifar10 --batch_size 32 --start_revision 29 --task classification --threshold 0.3
```

Running the Scalar Random Dropout (SRD): 
```
python .\main.py --model mobilenet_v2 --mode train_with_percentage --epoch 30 --save_path cifar10_results/mobilenet_v2 --dataset cifar10 --batch_size 32 --start_revision 29 --task classification --threshold 0.3
```

Running the Selective Recall of Forgotten Items (SRFI):
```
python .\main.py --model mobilenet_v2 --mode train_with_selective_recall --epoch 30 --save_path cifar10_results/mobilenet_v2 --dataset cifar10 --batch_size 32 --start_revision 29 --task classification --threshold 0.3 --review_interval 5
```
The `--review_interval` flag controls how often (in epochs) previously dropped samples are re-evaluated and reintroduced if the model's confidence has degraded below the threshold. This mimics the cognitive principle of retrieval practice.

If you wish to change the percentage parameter, head to the train_with_random function and change the decay parameter.

Note: The current codebase provides a recipe for single GPU training, multi-GPU training codes will be released soon.

---

## Selective Recall Feature: Detailed Explanation of Changes

The [Selective Recall of "Forgotten" Items](https://github.com/Harshith325/data_dropout/commit/cea1f4a0a0201cf3ca5eabad9e8991809f5686c3) feature introduces a new data dropout technique across two files: `training_models/main.py` and `training_models/selective_gradient.py`.

### Changes to `training_models/main.py`

1. **New training mode registered**: The `--mode` argument's choices list was extended to include `"train_with_selective_recall"`, making it available as a CLI option.

2. **New CLI argument `--review_interval`**: A new command-line argument `--review_interval` (integer, default `5`) was added. This controls how frequently (in epochs) the method reviews previously dropped samples to check for forgotten ones.

3. **New dispatch branch**: A new `elif` block was added in the training mode dispatcher. When `--mode train_with_selective_recall` is selected, it:
   - Creates a `TrainRevision` instance with the standard parameters.
   - Prints the training mode, start revision epoch, and review interval.
   - Calls `train_revision.train_with_selective_recall(args.start_revision, args.task, cls_num_list, args.review_interval)`.
   - Prints the total number of training steps.

### Changes to `training_models/selective_gradient.py`

A new method `train_with_selective_recall(self, start_revision, task, cls_num_list, review_interval)` (~295 lines) was added to the `TrainRevision` class. Here is how it works:

#### Setup
- Configures the loss function: `CrossEntropyLoss` for standard classification, or `FocalLoss` with class-rebalancing weights for long-tail tasks.
- Uses the `AdamW` optimizer (lr=3e-4) with a `StepLR` scheduler (gamma=0.98).
- Initializes tracking structures: `epoch_losses`, `epoch_accuracies`, `epoch_test_accuracies`, `epoch_test_losses`, `time_per_epoch`, `survival_log`, `label_log`, `samples_used_per_epoch`.
- Creates an empty `dropped_indices` set to track which training sample indices have been confidently classified and subsequently dropped.

#### Training Loop — Dropout Phase (`epoch < start_revision`)
For each epoch before the revision threshold:

1. **Periodic Review of Dropped Samples**: Every `review_interval` epochs (when `dropped_indices` is non-empty), the method:
   - Switches the model to eval mode.
   - Creates a `Subset` data loader from only the dropped sample indices.
   - Runs inference on each dropped sample and computes the softmax probability for the correct class.
   - If the confidence falls below `self.threshold`, the sample is considered "forgotten" and its index is added to a `recalled_indices` list.
   - Recalled indices are removed from `dropped_indices`.
   - The model is switched back to train mode and the recalled (forgotten) samples are trained on in a mini training loop.

2. **Standard Dropout Training**: For each batch:
   - Runs a forward pass in `no_grad` mode to evaluate which samples the model is confident about.
   - If `threshold == 0`: keeps only misclassified samples. Otherwise: keeps only samples whose correct-class probability is below the threshold.
   - Easy/confident samples are added to `dropped_indices`; samples that have become hard again are removed from `dropped_indices`.
   - Only the hard/uncertain samples (`inputs_misclassified`) are used for the actual gradient update.
   - Tracks survival logs (which sample indices were used) and label distribution logs.

3. **Epoch Evaluation**: After each dropout-phase epoch, the model is evaluated on the test set, recording test accuracy and loss. The learning rate scheduler steps based on validation loss.

#### Training Loop — Full Review Phase (`epoch >= start_revision`)
After the `start_revision` epoch, the method switches to training on the **entire** dataset (no dropout), similar to the existing `train_with_revision` method. All samples are used for gradient updates, and test evaluation is performed each epoch.

#### Post-training
- Logs memory usage and total wall time.
- Generates accuracy-vs-time plots for both train and test metrics using the project's existing plotting utilities.
- Saves `survival_log_selective_recall.json` (which sample indices were trained on each epoch) and `label_log_selective_recall.json` (cumulative label distribution of trained samples).
- Returns the trained model and total number of training steps.

### Key Difference from Existing Methods
The core novelty compared to the existing `train_with_revision` method is the **periodic review mechanism**: instead of permanently dropping confident samples, the method revisits them every `review_interval` epochs and reintroduces any whose model confidence has degraded. This is inspired by the cognitive science principle of **retrieval practice** — the idea that revisiting previously learned material strengthens long-term retention.

---

## Summary (100 words)

The latest commit adds a new **Selective Recall of Forgotten Items** data dropout technique. During training, samples the model confidently classifies are progressively dropped to save computation. Every few epochs (controlled by `--review_interval`), all dropped samples are re-evaluated; any whose model confidence has fallen below the threshold are reintroduced for further training. This mimics the cognitive principle of retrieval practice—revisiting forgotten material to reinforce learning. The implementation adds a `train_with_selective_recall` method to `selective_gradient.py` (~295 lines) and integrates it into `main.py` with a new CLI argument. After the dropout phase, a full-dataset review phase follows.
