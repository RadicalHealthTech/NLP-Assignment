import os

working_dir = os.getcwd()

json_file = os.path.join(working_dir, "data/train-v2.0.json")
json_test_file = os.path.join(working_dir, "data/dev-v2.0.json")
vocab_file = os.path.join(working_dir, "data/vocab")
load_model = os.path.join(working_dir, "saved_model.pt")
sample = True

BATCH_SIZE = 128
ENC_EMB_DIM = 50
DEC_EMB_DIM = 50
HID_DIM = 50
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
N_EPOCHS = 20
CLIP = 1
