## Sequential Feature Selection for Neural Rankers in Learning to Rank

## Experimental Setup and Execution

### Prerequisites
Ensure all required dependencies are installed.

### Model Training
To train models, execute the training phase using the following command:

```bash
python train.py models=sequential_selector \
                 datasets=MQ2008 \
                 train_mode=listwise \
                 device=mps \
                 random_seed=111
```

### Model Evaluation
For evaluation of the trained model, execute:

```bash
python test.py models=sequential_selector \
                datasets=MQ2008 \
                train_mode=listwise \
                device=mps \
                random_seed=111
```

