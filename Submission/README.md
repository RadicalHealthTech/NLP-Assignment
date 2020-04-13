# Natural Language Processing Assignment

In this assignment, we will be working on the problem of Question Answering (QA) using NLP.  
We are using the Sequence to Sequence Encoder Decoder model with GRU RNN units.

### Folder structure:
```
├── data
│   ├── dev-v2.0.json
│   └── train-v2.0.json
├── env
├── README.md
├── requirements.txt
└── src
    ├── config.py
    ├── model.py
    ├── squad_data.py
    └── trainer.py
```
### Setup:

1. Install Python-3.7
1. Install required packages using ```pip install -r requirements.txt```

### Run:
1. Set the hyperparametes in config.py. For testing on a sample, use Sample=True.
1. ```python src/trainer.py```

### TODO
- Add hyperparameter search: ray[tune], add sample config_hp_tun.py

