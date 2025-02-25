import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from encoding import Encode


class LogDataset(Dataset):
    """Custom PyTorch dataset for log anomaly detection."""

    def __init__(self, sequence, sequence_label, key_label):
        self.sequence = sequence
        self.sequence_label = sequence_label
        self.key_label = key_label

    def __len__(self):
        return len(self.sequence_label)

    def __getitem__(self, idx):
        return self.sequence[idx], self.sequence_label[idx], self.key_label[idx]


class DataProcessor:
    def __init__(self, dataset_path, batch_train, batch_test, batch_val, batch_train_test):
        self.dataset_path = dataset_path
        self.batch_sizes = {
            'train': batch_train,
            'test': batch_test,
            'val': batch_val,
            'train_test': batch_train_test
        }

        # Load and preprocess dataset
        dataset = self.load_and_preprocess_data(self.dataset_path)

        # Split dataset
        self.train_ds, self.test_ds, self.val_ds = self.split_data(dataset)

        # Build log event mapping
        self.logkey2index, self.logkeys = Encode.build_logkey_mapping(self.train_ds)
        self.vocab_size = len(self.logkeys)

        # Encode datasets
        train_data, test_data, val_data = Encode.encode_datasets(
            self.train_ds, self.test_ds, self.val_ds, self.logkey2index
        )

        # Prepare data loaders
        self.train_loader = self.dataset_dataloader(train_data, self.batch_sizes['train'])
        self.test_loader = self.dataset_dataloader(test_data, self.batch_sizes['test'])
        self.val_loader = self.dataset_dataloader(val_data, self.batch_sizes['val'])

    @staticmethod
    def load_and_preprocess_data(filepath):
        """Load dataset and apply sliding window preprocessing."""
        logdata = pd.read_csv(filepath)
        return DataProcessor.slide_window(logdata)

    @staticmethod
    def slide_window(logdata, window_size=20, step_size=10):
        """Apply sliding window approach to log data."""
        logdata["Label"] = logdata["Label"].apply(lambda x: int(x != '-'))
        data = logdata.loc[:, ['EventId', 'Label']]
        data['Key_label'] = data['Label']
        data.rename(columns={'Label': 'Sequence_label'}, inplace=True)

        new_data = []
        for idx in range(0, len(data) - window_size + 1, step_size):
            new_data.append([
                data['EventId'].iloc[idx: idx + window_size].values,
                max(data['Key_label'].iloc[idx: idx + window_size]),
                data['Key_label'].iloc[idx: idx + window_size].values
            ])

        return pd.DataFrame(new_data, columns=['EventId', 'Sequence_label', 'Key_label'])

    @staticmethod
    def split_data(dataset):
        """Split dataset into training, validation, and test sets."""
        normal_ds = dataset[dataset['Sequence_label'] == 0]
        abnormal_ds = dataset[dataset['Sequence_label'] == 1]

        train_ds, rest_ds = train_test_split(normal_ds, test_size=0.2, random_state=2021)
        test_normal_ds, val_normal_ds = train_test_split(rest_ds, test_size=0.1, random_state=2021)
        test_abnormal_ds, val_abnormal_ds = train_test_split(abnormal_ds, test_size=0.1, random_state=2021)

        test_ds = pd.concat([test_normal_ds, test_abnormal_ds])
        val_ds = pd.concat([val_normal_ds, val_abnormal_ds])

        return train_ds, test_ds, val_ds

    @staticmethod
    def dataset_dataloader(data, batch_size):
        """Prepare PyTorch dataloaders."""
        dataset = LogDataset(
            sequence=data['Encoded'].tolist(),
            sequence_label=data['Sequence_label'].tolist(),
            key_label=data['Key_label'].tolist()
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
