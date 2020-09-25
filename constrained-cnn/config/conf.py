import os, torch, random
random.seed(100)
torch.manual_seed(100)
torch.cuda.manual_seed(100)

class Conf(object):
    ROOT = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    total_class = 2
    total_epoch = 1000
    batch_size = 128
    learning_rate = 0.01 # initial
    # learning_rate = 0.00018 # parameter2020-09-24 20-17-26.pkl

    data_path = os.path.join(ROOT, "dataset")
    model_path = os.path.join(ROOT, "model")