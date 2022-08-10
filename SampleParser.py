

class SampleParser:
    def __init__(self, tokenizer, labels=None, padding_len=None):
        self.tokenizer = tokenizer
        if labels is None:
            self.labels = {'O', 'B-purchaser', 'B-material', 'B-service', 'B-engineer',
                           'I-purchaser', 'I-material', 'I-service', 'I-engineer',
                           'E-purchaser', 'E-material', 'E-service', 'E-engineer',
                           'S-purchaser', 'S-material', 'S-service', 'S-engineer'}
        else:
            self.labels = labels
        self.labels_to_ids = {label: i for i, label in enumerate(self.labels)}
        self.ids_to_labels = {i: label for label, i in self.labels_to_ids.items()}
        if padding_len is None:
            self.padding_len = -1
        else:
            self.padding_len = padding_len

    def parse_sample(self, text, tags):
        tokens = ['[CLS]']
        for c in text:
            tokens.append(c)
        # text_tokens = self.tokenizer.tokenize(text)
        # tokens.extend(text_tokens)
        tokens.append('[SEP]')
        attention_mask = [1 for _ in range(len(tokens))]
        token_type_ids = [0 for _ in range(len(tokens))]
        tags.insert(0, 'O')
        tags.append('O')
        d = len(tags)
        assert len(tags) == len(tokens)
        while len(tokens) < self.padding_len:
            tokens.append('[PAD]')
            attention_mask.append(0)
            token_type_ids.append(1)
            tags.append('O')
        return {
            'input_ids': self.tokenizer.convert_tokens_to_ids(tokens),
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'tags': [self.labels_to_ids[i] for i in tags]
        }

    def set_pad_len(self, pl):
        self.padding_len = pl

    def decode(self, text, label_seq):
        label_seq = [self.ids_to_labels[_] for _ in label_seq]
        start_idx = 0
        now_attr = ''
        attr_dict = dict()
        for i in range(len(text)):
            if label_seq[i] == 'O':
                continue
            if label_seq[i][0] == 'S':
                attr_dict[text[i: i+1]] = (i, i+1, label_seq[i][2:])
            if label_seq[i][0] == 'B':
                start_idx = i
                now_attr = label_seq[i][2:]
            if label_seq[i][0] == 'I':
                if not label_seq[i][2:] == now_attr:
                    start_idx = i
                    now_attr = ''
                continue
            if label_seq[i][0] == 'E':
                if now_attr == label_seq[i][2:]:
                    attr_dict[text[start_idx: i+1]] = (start_idx, i+1, now_attr)
        return attr_dict
