from torch.optim import Adagrad, optimizer
import torch
from typing import List
from SampleParser import SampleParser
from use_config import config
from tqdm import tqdm


def train(train_set, valid_set, model, device, parser: SampleParser, n_epoch,
          save_model_path):
    device = torch.device(device)
    model = model.to(device)
    max_f1 = 0
    optimizer_bert = Adagrad(model.bert_model.parameters(), config.bert_lr, config.bert_lr_decay)
    optimizer_head = Adagrad(model.project.parameters(), config.head_lr, config.head_lr_decay)
    optimizer_crf = Adagrad(model.crf.parameters(), config.head_lr, config.head_lr_decay)
    optimizers = [optimizer_bert, optimizer_head, optimizer_crf]
    train_loader, train_batches = train_set.get_loader()
    min_loss = 10**9
    stop_cnt = 0
    for epoch in range(n_epoch):
        train_loss = 0.
        model.train()
        for i, batch in tqdm(enumerate(train_loader), total=train_batches):
            model.zero_grad()
            # for op in optimizers:
            #     op.zero_grad()
            inputs = {
                'input_ids': torch.LongTensor(batch['input_ids']).to(device),
                'attention_mask': torch.LongTensor(batch['attention_mask']).to(device),
                'token_type_ids': torch.LongTensor(batch['token_type_ids']).to(device)
            }
            tags = torch.LongTensor(batch['tags']).to(device)
            score = -model(tags, **inputs)
            score.backward()
            for op in optimizers:
                op.step()
            train_loss = (train_loss * i / (i+1)) + (score / (i+1))
        improved = ''
        entities, predicted_entities, correct_predicts, span_corr, val_loss = valid(valid_set, model, device, parser)
        recall = predicted_entities / (entities + 1e-5)
        precise = correct_predicts / (predicted_entities + 1e-5)
        accuracy = correct_predicts / (entities + 1e-5)
        f1 = 2/(1/(precise+1e-5) + 1/(recall+1e-5))
        if val_loss < min_loss:
            stop_cnt = 0
            min_loss = val_loss
            improved = '*'
            torch.save(model.state_dict(), save_model_path)
        else:
            stop_cnt += 1
            if stop_cnt == config.pre_stop:
                print('{}个epoch未改进, 已提前停止, 最优loss为{}'.format(config.pre_stop, min_loss))
        print('{}个epoch, 共{}实体, 预测出{}实体, 预测正确{}实体, 属性错误但位置正确{}'.
              format(epoch, entities, predicted_entities, correct_predicts, span_corr))
        print('acc:{:.4f}, pre:{:.4f}, rec:{:.4f}, f1:{:.4f}'.format(accuracy, precise, recall, f1))
        print('train_loss: {:.4f}, val_loss: {:.4f} {}'.format(train_loss, val_loss, improved))
    return min_loss


def valid(val_set, model, device, parser):
    val_loader, val_batches = val_set.get_loader()
    model.eval()
    val_loss = 0.
    with torch.no_grad():
        total_attrs = 0
        predicted_attrs = 0
        correct_attrs = 0
        span_corr = 0
        for i, batch in tqdm(enumerate(val_loader), total=val_batches):
            inputs = {
                'input_ids': torch.LongTensor(batch['input_ids']).to(device),
                'attention_mask': torch.LongTensor(batch['attention_mask']).to(device),
                'token_type_ids': torch.LongTensor(batch['token_type_ids']).to(device)
            }
            out_tags, loss = model.inference(**inputs)
            val_loss = (val_loss * i) / (i+1) + (-loss / (i+1))
            for j in range(out_tags.shape[0]):
                true_attr = parser.decode(batch['text'][j], batch['tags'][j][1:])
                predicted_attr = parser.decode(batch['text'][j], out_tags[j].cpu().numpy()[1:])
                total_attrs += len(true_attr)
                predicted_attrs += len(predicted_attr)
                for word, tri in predicted_attr.items():
                    if word in true_attr:
                        if true_attr[word][0] == tri[0] and true_attr[word][1] == tri[1]\
                                and true_attr[word][2] == tri[2]:
                            correct_attrs += 1
                        else:
                            span_corr += 1

        return total_attrs, predicted_attrs, correct_attrs, span_corr, val_loss

