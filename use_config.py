import argparse

config = argparse.Namespace(
    data_path='data/labeling',
    model_name='bert-base-chinese',
    pad_len=256,
    batch_size=8,
    bert_lr=1e-5,
    bert_lr_decay=1e-7,
    head_lr=1e-3,
    head_lr_decay=1e-5,
    device='cuda',
    n_epoch=100,
    save_model_path='model/saved_model' + '_zc0809.pkl',
    pre_stop=10
)
