import numpy as np
from collections import Counter


class Encode:
    def __init__(self):
        self.logkey2index, self.logkeys = None, None

    @staticmethod
    def build_logkey_mapping(train_ds):
        """Create a mapping from log events to indices."""
        counts = Counter()
        for _, row in train_ds.iterrows():
            counts.update(row['EventId'])

        logkey2index = {"": 0, "UNK": 1}
        logkeys = ["", "UNK"]

        for word in counts:
            logkey2index[word] = len(logkeys)
            logkeys.append(word)

        return logkey2index, logkeys

    @staticmethod
    def encode_sequence(sequence, logkey2index):
        """Encode event sequences using the logkey mapping."""
        return np.array([logkey2index.get(logkey, logkey2index["UNK"]) for logkey in sequence])

    @staticmethod
    def encode_datasets(train_ds, test_ds, val_ds, logkey2index):
        """Encode datasets using log event mapping."""
        for ds in [train_ds, test_ds, val_ds]:
            ds['Encoded'] = ds['EventId'].apply(lambda seq: Encode.encode_sequence(seq, logkey2index))

        return (
            train_ds[['Encoded', 'Sequence_label', 'Key_label']],
            test_ds[['Encoded', 'Sequence_label', 'Key_label']],
            val_ds[['Encoded', 'Sequence_label', 'Key_label']]
        )
