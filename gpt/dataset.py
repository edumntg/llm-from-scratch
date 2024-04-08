import torch
from torch.utils.data import Dataset

class GPTDataset(Dataset):
    """Receives raw text and tokenizer
    Computes all IDs for text"""
    def __init__(self, text, tokenizer, max_length, stride):
        super().__init__()

        self.tokenizer = tokenizer
        self.inputs = []
        self.labels = []

        token_ids = self.tokenizer.encode(text)
        for i in range(0, len(token_ids) - max_length, stride):
            X = token_ids[i:i+max_length]
            y = token_ids[i+1:i+max_length+1]
            self.inputs.append(torch.tensor(X))
            self.labels.append(torch.tensor(y))
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]