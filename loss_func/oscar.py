# we have a binary classifier that predicts whether the 
# pair is corrupted or not

from torch.nn import CrossEntropyLoss

def get_loss(**kwargs):
    # get logits, labels from the kwargs
    loss = CrossEntropyLoss()
    return loss(kwargs["logits"], kwargs["labels"])