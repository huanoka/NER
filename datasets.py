from torch.utils.data import DataLoader, Dataset
import pandas as pd
from use_config import config


class SequenceLabelDataset(Dataset):
    def __init__(self, data_path, split, parser):
        if split not in ['train', 'test', 'dev']:
            raise ValueError(f'split must in train test or dev, got {split}')
        self.df = pd.read_csv(data_path + '/' + str(split) + '_split', header=0)
        self.parser = parser

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        line = self.df.iloc[index]
        text = line['text']
        labels = line['labels']

        return {
            'text': text,
            'tags': labels.split(' ')
        }

    def collate_fn(self, batch):
        token_ids_batch = []
        attention_mask_batch = []
        token_type_ids_batch = []
        tags_batch = []
        texts = []
        pad_flag = self.parser.padding_len == -1
        if pad_flag:
            self.parser.set_pad_len(max([len(_['text']) for _ in batch]))
        for b in batch:
            text = b['text']
            tags = b['tags']
            r_dict = self.parser.parse_sample(text, tags)
            token_ids_batch.append(r_dict['input_ids'])
            attention_mask_batch.append(r_dict['attention_mask'])
            token_type_ids_batch.append(r_dict['token_type_ids'])
            tags_batch.append(r_dict['tags'])
            texts.append(text)
        if pad_flag:
            self.parser.set_pad_len(-1)
        return {
            'input_ids': token_ids_batch,
            'attention_mask': attention_mask_batch,
            'token_type_ids': token_type_ids_batch,
            'tags': tags_batch,
            'text': texts
        }

    def get_loader(self):
        loader = DataLoader(self, config.batch_size, shuffle=True, drop_last=False, collate_fn=self.collate_fn)
        max_batches = (len(self) // config.batch_size) + (0 if len(self) % config.batch_size == 0 else 1)
        return loader, max_batches





