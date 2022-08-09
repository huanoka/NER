from typing import Optional, List

import torch
import torch.nn as nn
from torch.nn.modules import LSTMCell
from transformers import BertTokenizer, BertModel


class NERModel(nn.Module):
    def __init__(self, config):
        super(NERModel).__init__()


class LSTMLayer(nn.Module):
    # Ct: Cell_state: 不会输出， 记录cell记忆的东西
    # Ht: Hidden State: 本cell输出的hidden_state
    # Xt: 当前输入的向量
    # Xt: [B * EMB_D] Ht: [B * HID_D]  ==> Wf:[ , emb_d + hid_d]    Ft: []
    # Ct: []
    # W_f W_i, W_c, W_v: [emb_d + hid_d, hid_d]

    def __init__(self, e_dim, h_dim, sigma):
        super(LSTMLayer).__init__()
        self.W_f = nn.Parameter(torch.rand(e_dim + h_dim, h_dim))
        self.b_f = nn.Parameter(torch.rand(h_dim))
        self.W_i = nn.Parameter(torch.rand(e_dim + h_dim, h_dim))
        self.b_i = nn.Parameter(torch.rand(h_dim))
        self.W_c = nn.Parameter(torch.rand(e_dim + h_dim, h_dim))
        self.b_c = nn.Parameter(torch.rand(h_dim))
        self.W_o = nn.Parameter(torch.rand(e_dim + h_dim, h_dim))
        self.b_o = nn.Parameter(torch.rand(h_dim))
        self.c = torch.rand(h_dim)
        self.sigma = sigma

    def forward(self, x_input: torch.Tensor):
        c_pre = self.c
        concated = torch.concat((x_input, c_pre))
        f = self.sigma * (torch.mul(concated, self.W_f) + self.b_f)
        i = self.sigma * (torch.mul(concated, self.W_i) + self.b_i)
        c = torch.tanh_(torch.mul(concated, self.W_c) + self.b_c)
        o = self.sigma * (torch.mul(concated, self.W_o) + self.b_o)
        c = torch.mul(f, c_pre) + torch.mul(i, c)
        self.c = c
        h_out = torch.mul(o, torch.tanh_(c))
        return h_out


class CRFLayer(nn.Module):
    def __init__(self, num_tags: int, batch_first: bool = False) -> None:
        super(CRFLayer, self).__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_tags={self.num_tags})'

    def forward(self, emissions: torch.Tensor,
                tags: torch.LongTensor,
                mask: Optional[torch.ByteTensor] = None,
                reduction: str = 'mean') -> torch.Tensor:
        if reduction not in ('none', 'sum', 'mean', 'token_mean'):
            raise ValueError(f'invalid reduction: {reduction}')
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.bool, device=tags.device)
        if mask.dtype != torch.uint8:
            mask = mask.byte()
        self._validate(emissions, tags=tags, mask=mask)
        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        numerator = self._compute_score(emissions, tags, mask)
        denominator = self._compute_normalizer(emissions, mask)
        llh = numerator - denominator
        if reduction == 'none':
            return llh
        if reduction == 'sum':
            return llh.sum()
        if reduction == 'mean':
            return llh.mean()
        return llh.sum() / mask.float().sum()

    def decode(self, emissions: torch.LongTensor,
               mask: Optional[torch.ByteTensor] = None,
               nbest: Optional[int] = None,
               pad_tag: Optional[int] = None) -> List[List[List[int]]]:
        if nbest is None:
            nbest = 1
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.uint8, device=emissions.device)
        if mask.dtype != torch.uint8:
            mask = mask.byte()
        self._validate(emissions, mask)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)
        if nbest == 1:
            return self._viterbi_decode(emissions, mask, pad_tag).unsqueeze(0)
        return self._viterbi_decode_nbest(emissions, mask, nbest, pad_tag)

    def _validate(self, emissions:torch.Tensor, tags: Optional[torch.LongTensor],
                  mask: Optional[torch.ByteTensor] = None) -> None:
        if emissions.dim() != 3:
            raise ValueError(f'emissions must have dimension of 3, got {emissions.dim}')
        if emissions.size(2) != self.num_tags:
            raise ValueError(
                'the first two dimensions of emissions and tags must match,'
                f'got {tuple(emissions.shape[:2] and {tuple(tags.shape)})}'
            )
        if mask is not None:
            if emissions.shape[:2] != mask.shape:
                raise ValueError(
                    'the first two dimensions of emissions and mask must match,'
                    f'got {tuple(emissions.shape[:2] and {tuple(tags.shape)})}'
                )
            no_empty_seq = not self.batch_first and mask[0].all()
            no_empty_seq_bf = self.batch_first and mask[:, 0].all()
            if not no_empty_seq and not no_empty_seq_bf:
                raise ValueError('mask of the first timestep must all be on')

    def _compute_score(self, emissions: torch.Tensor,
                       tags: torch.LongTensor,
                       mask: torch.ByteTensor) -> torch.Tensor:
        seq_len, batch_size = tags.shape
        mask = mask.float()
        score = self.start_transitions[tags[0]]
        score += emissions[0, torch.arange(batch_size), tags[0]]

        for i in range(1, seq_len):
            score += self.transitions[tags[i - 1], tags[i]] * mask[i]
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]
        seq_ends = mask.long().sum(dim=0) - 1
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        score += self.end_transitions[last_tags]
        return score

    def _compute_normalizer(self, emissions: torch.Tensor, mask: torch.ByteTensor) -> torch.Tensor:
        seq_len = emissions.size(0)
        score = self.start_transitions + emissions[0]
        for i in range(1, seq_len):
            broadcast_score = score.unsqueeze(2)
            broadcast_emissions = emissions[i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emissions
            next_score = torch.logsumexp(next_score, dim=1)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
        score += self.end_transitions
        return torch.logsumexp(score, dim=1)

    def _viterbi_decode(self, emissions: torch.FloatTensor,
                        mask: torch.ByteTensor,
                        pad_tag: Optional[int] = None) -> List[List[int]]:
        if pad_tag is None:
            pad_tag = 0
        device = emissions.device
        seq_len, batch_size = mask.shape
        score = self.start_transitions + emissions[0]
        history_idx = torch.zeros((seq_len, batch_size, self.num_tags), dtype=torch.long, device=device)
        oor_idx = torch.zeros((batch_size, self.num_tags), dtype=torch.long, device=device)
        oor_tag = torch.full((seq_len, batch_size), pad_tag, dtype=torch.long, device=device)
        for i in range(1, seq_len):
            broadcast_score = score.unsqueeze(2)
            broadcast_emission = emissions[i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emission
            next_score, indices = next_score.max(dim=1)
            score = torch.where(mask[i].unsqueeze(-1), next_score, score)
            indices = torch.where(mask[i].unsqueeze(-1), indices, oor_idx)
            history_idx[i-1] = indices
        end_score = score + self.end_transitions
        _, end_tag = end_score.max(dim=1)
        seq_ends = mask.long().sum(dim=0) - 1
        history_idx = history_idx.transpose(1, 0).contiguous()
        dd = seq_ends.view(-1, 1, 1).expand(-1, 1, self.num_tags)
        ddd = end_tag.view(-1, 1, 1).expand(-1, 1, self.num_tags)
        history_idx.scatter_(1, seq_ends.view(-1, 1, 1).expand(-1, 1, self.num_tags),
                             end_tag.view(-1, 1, 1).expand(-1, 1, self.num_tags))
        history_idx = history_idx.transpose(1, 0).contiguous()
        best_tags_arr = torch.zeros((seq_len, batch_size), dtype=torch.long, device=device)
        best_tags = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        for idx in range(seq_len - 1, -1, -1):
            best_tags = torch.gather(history_idx[idx], 1, best_tags)
            best_tags_arr[idx] = best_tags.data.view(batch_size)
        return torch.where(mask, best_tags_arr, oor_tag).transpose(0, 1)

    def _viterbi_decode_nbest(self, emissions: torch.FloatTensor, mask: torch.ByteTensor, nbest: int,
                              pad_tag: Optional[int] = None) -> List[List[List[int]]]:
        if pad_tag is None:
            pad_tag = 0
        device = emissions.device
        seq_length, batch_size = mask.shape
        score = self.start_transitions + emissions[0]
        history_idx = torch.zeros((seq_length, batch_size, self.num_tags, nbest), dtype=torch.long, device=device)
        oor_idx = torch.zeros((batch_size, self.num_tags, nbest), dtype=torch.long, device=device)
        oor_tag = torch.full((seq_length, batch_size, nbest), pad_tag, dtype=torch.long, device=device)
        for i in range(1, seq_length):
            if i == 1:
                broadcast_score = score.unsqueeze(-1)
                broadcast_emission = emissions[i].unsqueeze(1)
                next_score = broadcast_score + self.transitions + broadcast_emission
            else:
                broadcast_score = score.unsqueeze(-1)
                broadcast_emission = emissions[i].unsqueeze(1).unsqueeze(2)
                next_score = broadcast_score + self.transitions.unsqueeze(1) + broadcast_emission
                next_score, indices = next_score.view(batch_size, -1, self.num_tags).topk(nbest, dim=1)
            next_score, indices = next_score.view(batch_size, -1, self.num_tags).topk(nbest, dim=1)
            if i == 1:
                score = score.unsqueeze(-1).expand(-1, -1, nbest)
                indices = indices * nbest
            next_score = next_score.transpose(2, 1)
            indices = indices.transpose(2, 1)
            score = torch.where(mask[i].unsqueeze(-1).unsqueeze(-1), next_score, score)
            indices = torch.where(mask[i].unsqueeze(-1).unsqueeze(-1), indices, oor_idx)
            history_idx[i-1] = indices

        end_score = score + self.end_transitions.unsqueeze(-1)
        _, end_tag = end_score.view(batch_size, -1).topk(nbest, dim=-1)
        seq_ends = mask.long().sum(dim=0) - 1
        history_idx = history_idx.transpose(1, 0).contiguous()
        history_idx.scatter_(1, seq_ends.view(-1, 1, 1, 1).expand(-1, 1, self.num_tags, nbest),
                             end_tag.view(-1, 1, 1, nbest).expand(-1, 1, self.num_tags, nbest))
        history_idx = history_idx.transpose(1, 0).contiguous()
        best_tags_arr = torch.zeros((seq_length, batch_size, nbest), dtype=torch.long, device=device)
        best_tags = torch.arange(nbest, dtype=torch.long, device=device) \
            .view(1, -1).expand(batch_size, -1)
        for idx in range(seq_length - 1, -1, -1):
            best_tags = torch.gather(history_idx[idx].view(batch_size, -1), 1, best_tags)
            best_tags_arr[idx] = best_tags.data.view(batch_size, -1) // nbest

        return torch.where(mask.unsqueeze(-1), best_tags_arr, oor_tag).permute(2, 1, 0)


class BertCRF(nn.Module):
    def __init__(self, bert_model_name, num_classes, dropout_prob):
        super(BertCRF, self).__init__()
        self.bert_model = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout_prob)
        self.project = nn.Linear(768, num_classes)
        self.crf = CRFLayer(num_classes, batch_first=True)

    def forward(self, input_tags, **kwargs):
        bert_out = self.bert_model(**kwargs)
        word_hiddens = bert_out.last_hidden_state
        word_hiddens = self.dropout(word_hiddens)
        emissions = self.project(word_hiddens)
        score = self.crf.forward(emissions, input_tags, kwargs['attention_mask'], 'token_mean')
        return score

    def inference(self, **kwargs):
        bert_out = self.bert_model(**kwargs)
        word_hiddens = bert_out.last_hidden_state
        emissions = self.project(word_hiddens)
        out_tags = self.crf.decode(emissions, kwargs['attention_mask'])[0]
        score = self.crf.forward(emissions, out_tags, kwargs['attention_mask'], 'token_mean')
        return out_tags, score


class BertLSTMCRF(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BertLSTMCRF, self).__init__()
        self.bert_model = BertModel.from_pretrained(bert_model_name)
        self.project = nn.Linear(768, num_classes),
        self.crf = CRFLayer(num_classes, batch_first=True)

    def forward(self, input_tags, **kwargs):
        bert_out = self.bert_model(**kwargs)
        word_hiddens = bert_out.last_hidden_state
        emissions = self.project(word_hiddens)
        score = self.crf.forward(emissions, input_tags, kwargs['attention_mask'], 'token_mean')
        return score

    def inference(self, **kwargs):
        bert_out = self.bert_model(**kwargs)
        word_hiddens = bert_out.last_hidden_state
        emissions = self.project(word_hiddens)
        out_tags = self.crf.decode(emissions, kwargs['attention_mask'])
        return out_tags

def test():
    model = BertCRF('bert-base-chinese', 3)
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    sen = '夜宵才是生命的真谛'
    text_tokens = tokenizer.tokenize(sen)
    tokens = ['[CLS]']
    tokens.extend(text_tokens)
    tokens.append('[SEP]')
    labels = [2, 0, 1, 2, 2, 2, 2, 2, 0, 1, 2]
    attention_mask = [1 for _ in range(len(tokens))]
    token_type_ids = [0 for _ in range(len(tokens))]
    while len(tokens) < 16:
        tokens.append('[PAD]')
        attention_mask.append(0)
        token_type_ids.append(1)
        labels.append(2)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    labels = torch.LongTensor([labels])
    inputs = {
        'input_ids': torch.LongTensor([token_ids]),
        'attention_mask': torch.LongTensor([attention_mask]),
        'token_type_ids': torch.LongTensor([token_type_ids])
    }
    score = model.forward(labels, **inputs)
    out_labels = model.inference(**inputs)
    print(score)
    print(out_labels)


