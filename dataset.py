from torch.utils.data import Dataset, DataLoader


class TrainDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=256):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class TestDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=256):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]