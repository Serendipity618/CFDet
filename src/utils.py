import torch
import random
import numpy as np
import pandas as pd
from sklearn import metrics
from preprocessing import DataProcessor


def setup_seed(seed=10):
    """Set the random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def find_baseline_sequence(model, center, radius, data_loader, device):
    seq_list, distance_list = [], []
    model.eval()

    with torch.no_grad():
        for sequence, _, _ in data_loader:
            seq_list.extend(sequence.tolist())
            sequence = sequence.to(device)
            hidden = model(sequence)
            distance = torch.mean(torch.square(hidden - center), dim=1)
            distance_list.extend(distance.tolist())

    return torch.tensor(seq_list[np.argmin(distance_list)]).to(device)


def split_sequences_by_distance(model, center, radius, test_loader, device):
    sequences = {"far": [], "medium": [], "close": []}
    labels = {"far": [], "medium": [], "close": []}
    keys = {"far": [], "medium": [], "close": []}

    model.eval()
    with torch.no_grad():
        for sequence, sequence_label, key_label in test_loader:
            sequence = sequence.to(device)
            hidden = model(sequence)
            distance = torch.mean(torch.square(hidden - center), dim=1)

            sequence_l, sequence_label_l, key_label_l = sequence.tolist(), sequence_label.tolist(), key_label.tolist()

            for i, d in enumerate(distance):
                if d > 10 * radius:
                    sequences["far"].append(sequence_l[i])
                    labels["far"].append(sequence_label_l[i])
                    keys["far"].append(key_label_l[i])
                elif d > radius:
                    sequences["medium"].append(sequence_l[i])
                    labels["medium"].append(sequence_label_l[i])
                    keys["medium"].append(key_label_l[i])
                else:
                    sequences["close"].append(sequence_l[i])
                    labels["close"].append(sequence_label_l[i])
                    keys["close"].append(key_label_l[i])

    return sequences, labels, keys


def train_test_data_loader(sequence_list, sequence_label_list, key_label_list, batch_size):
    df = pd.DataFrame({
        'Encoded': [torch.tensor(seq) for seq in sequence_list],
        'Sequence_label': [torch.tensor(l) for l in sequence_label_list],
        'Key_label': [torch.tensor(key) for key in key_label_list]
    })
    return DataProcessor.dataset_dataloader(df, batch_size), df
