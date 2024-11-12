import torch

IMG_SHAPE = (356, 356)
ROOT_DIR = 'flicker8k/images'
ANNOTATION_FILE = 'flicker8k/caption.txt'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LOAD_MODEL = False
SAVE_MODEL = True

# Hyper parameters
EMBED_SIZE = 256
HIDDEN_SIZE = 256
NUM_LAYERS = 1
LR = 3e-4
NUM_EPOCHS = 1000

CHECKPOINT = 'my_checkpoint.pth.tar'
