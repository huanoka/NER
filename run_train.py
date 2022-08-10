import torch

from use_config import config
from SampleParser import SampleParser
from transformers import BertTokenizer
from datasets import SequenceLabelDataset
from train import train, valid
from model import BertCRF
from torch.optim import Adagrad


def train_pipeline():
    tokenizer = BertTokenizer.from_pretrained(config.model_name)
    parser = SampleParser(tokenizer, None, getattr(config, 'pad_len', None))
    train_set = SequenceLabelDataset(config.data_path, 'train', parser)
    val_set = SequenceLabelDataset(config.data_path, 'dev', parser)
    model = BertCRF(config.model_name, 17, 0.5)
    min_loss = train(train_set, val_set, model, config.device, parser, config.n_epoch, config.save_model_path)
    print('after {} epoch, we got min_loss:{}'.format(config.n_epoch, min_loss))
    print('over!!!!!')


def test_pipeline():
    tokenizer = BertTokenizer.from_pretrained(config.model_name)
    parser = SampleParser(tokenizer, None, getattr(config, 'pad_len', None))
    test_set = SequenceLabelDataset(config.data_path, 'test', parser)
    model = BertCRF(config.model_name, 17, 0.5)
    model.load_state_dict(torch.load(config.save_model_path, map_location='cpu'))
    model = model.to(config.device)
    entities, predicted_entities, correct_predicts, span_corr, test_loss = valid(test_set, model, config.device, parser)
    recall = predicted_entities / (entities + 1e-5)
    precise = correct_predicts / (predicted_entities + 1e-5)
    accuracy = correct_predicts / (entities + 1e-5)
    f1 = 2 / (1 / (precise + 1e-5) + 1 / (recall + 1e-5))
    print('共{}实体, 预测出{}实体, 预测正确{}实体, 属性错误但位置正确{}'.
          format(entities, predicted_entities, correct_predicts, span_corr))
    print('acc:{:.4f}, pre:{:.4f}, rec:{:.4f}, f1:{:.4f}'.format(accuracy, precise, recall, f1))
    print('test_loss {:.4f}'.format(test_loss))


train_pipeline()
test_pipeline()
