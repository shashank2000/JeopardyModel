# find the negatives, then return loss
import torch

def get_loss(trip1, trip2, tripneg, tau):
    return torch.sum(trip1 * trip2) / tau - torch.sum(trip1 * tripneg) / tau
