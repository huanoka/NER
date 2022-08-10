import torch

from model import CRFLayer


def CRFtest():
    num_tags = 5
    crf = CRFLayer(num_tags)
    batch = 8
    seq_len = 12
    emissions = torch.randn((seq_len, batch, num_tags), dtype=torch.float)
    tags = torch.randint(0, num_tags, (seq_len, batch))
    # score = crf.forward(emissions, tags)
    # print(score)
    out = crf.decode(emissions, nbest=3)
    print(out)


CRFtest()
