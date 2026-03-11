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

Running the Adaptive Forgetting Schedule (AFS):
```
python .\main.py --model mobilenet_v2 --mode train_with_forgetting_schedule --epoch 30 --save_path cifar10_results/mobilenet_v2 --dataset cifar10 --batch_size 32 --start_revision 29 --task classification --threshold 0.3 --review_interval 5
```
Inspired by human spaced repetition (the spacing effect), this technique alternates between dropout epochs (where confident samples are dropped) and periodic "review" epochs (where the full dataset is revisited). The `--review_interval` flag controls how often (in epochs) a full-dataset review phase occurs during the dropout period. This helps the model consolidate learned material while still benefiting from progressive data dropout.

Running the Dynamic Difficulty Threshold (DDT):
```
python .\main.py --model mobilenet_v2 --mode train_with_dynamic_threshold --epoch 30 --save_path cifar10_results/mobilenet_v2 --dataset cifar10 --batch_size 32 --start_revision 29 --task classification --threshold 0.3 --initial_threshold 0.1 --max_threshold 0.9
```
Instead of a fixed confidence threshold for dropping samples, this technique uses a threshold that linearly increases over the dropout epochs from `--initial_threshold` to `--max_threshold`. Early on, only very easy (high confidence) samples are dropped; as training progresses and the model becomes more competent, moderately confident samples are also dropped. This mimics human learners who progressively consider more items "known" and focus only on what remains challenging.

Running the Cognitively-Motivated Data Ordering (CMDO):
```
python .\main.py --model mobilenet_v2 --mode train_with_cognitive_ordering --epoch 30 --save_path cifar10_results/mobilenet_v2 --dataset cifar10 --batch_size 32 --start_revision 29 --task classification --threshold 0.3 --curriculum_order curriculum
```
This technique orders training samples by difficulty (measured by the model's confidence) before applying progressive dropout. Use `--curriculum_order curriculum` to present easy samples first (high confidence) or `--curriculum_order anti_curriculum` to present hard samples first (low confidence). At each dropout epoch, the model's current confidence scores are used to re-sort the data, and then the standard confidence-based dropout is applied on the sorted batches. This reveals whether certain orderings can bootstrap the model better before dropout kicks in.

Running the Distribution-Preserving Sampling (DPS):
```
python .\main.py --model mobilenet_v2 --mode train_with_distribution_preserving --epoch 30 --save_path cifar10_results/mobilenet_v2 --dataset cifar10 --batch_size 32 --start_revision 29 --task classification --threshold 0.3
```
For each batch, the model computes confidence scores and splits samples into easy (confidence ≥ threshold) and hard (confidence < threshold) groups. Let the easy-to-hard ratio be x:y and the number of hard samples be z. Instead of training only on the z hard samples, this technique constructs a training subset of size z by sampling from both easy and hard samples while preserving the original x:y ratio. This introduces stochasticity while maintaining the overall difficulty distribution balance of the batch.

Running the Hard-Biased Sampling (HBS):
```
python .\main.py --model mobilenet_v2 --mode train_with_hard_biased --epoch 30 --save_path cifar10_results/mobilenet_v2 --dataset cifar10 --batch_size 32 --start_revision 29 --task classification --threshold 0.3
```
Similar to Distribution-Preserving Sampling, this technique splits each batch into easy and hard samples and constructs a training subset of size z. However, instead of preserving the original x:y ratio, the sampling proportions are reversed to y:x, meaning a larger fraction of the selected samples are drawn from the hard sample group. This creates a hardness-biased training subset that emphasizes difficult samples while still including some easier examples to stabilize learning.

If you wish to change the percentage parameter, head to the train_with_random function and change the decay parameter.

Note: The current codebase provides a recipe for single GPU training, multi-GPU training codes will be released soon.
