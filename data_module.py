from torchvision import transforms
from dataset.jeopardy_dataset import JeopardyDataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from torch import tensor
from PIL import ImageFilter
import random
import torch
import os

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""
    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = sigma
    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
        
class VQADataModule(LightningDataModule):
  def __init__(self, batch_size, threshold=10, num_workers=8, val_split=0.2, dumb_transfer=False, num_answers=0, transfer=False, multiple_images=False):
    super().__init__()
    self.batch_size = batch_size
    self.val_split = val_split
    self.test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.491, 0.482, 0.446],
                            std=[0.247, 0.243, 0.261]),
    ])
    
    if not transfer:
        # random crop, color jitter etc 
        self.train_transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                    transforms.RandomApply([
                        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                    ], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.RandomApply([GaussianBlur([1, 1])], p=0.5),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.491, 0.482, 0.446],
                            std=[0.247, 0.243, 0.261]),
                ])
    else:
        # for finetuning/linear eval - confirm
        self.train_transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.491, 0.482, 0.446],
                            std=[0.247, 0.243, 0.261]),
                ]
                )  
    
    # read env variables here
    self.questions_file = os.environ.get('QUESTIONS_FILE')
    self.answers_file = os.environ.get('ANSWERS_FILE')
    self.coco_loc = os.environ.get('COCO_LOC')   
    
    saved_train_file = 'train{}_correct_padding.pt'.format(self.is_dumb(dumb_transfer, num_answers, multiple_images))
    saved_test_file = 'test{}_correct_padding.pt'.format(self.is_dumb(dumb_transfer, num_answers, multiple_images))

    try:
        self.train_dataset = torch.load(saved_train_file)
        self.test_dataset = torch.load(saved_test_file)
        print("found files")
    except:
        print(saved_train_file + " not found")
        self.train_dataset = JeopardyDataset(
          self.questions_file, 
          self.answers_file, 
          self.coco_loc, 
          self.train_transform, num_answers=num_answers, 
          frequency_threshold=threshold, train=True, 
          dumb_transfer=dumb_transfer,
          multiple_images=multiple_images)
        self.test_dataset = JeopardyDataset(
          self.questions_file, 
          self.answers_file, 
          self.coco_loc, 
          self.test_transform, 
          frequency_threshold=threshold,
          word2idx=self.train_dataset.word2idx, 
          most_common_answers=self.train_dataset.most_common_answers, 
          dumb_transfer=dumb_transfer, 
          num_answers=num_answers, 
          train=False)
        
        torch.save(self.train_dataset, saved_train_file)
        torch.save(self.test_dataset, saved_test_file)

    self.num_workers = num_workers
    self.vl = self.get_vocab_length()
    self.idx_to_word = {v: k for k, v in self.train_dataset.word2idx.items()}
    if dumb_transfer:
      num_answer_classes = len(self.train_dataset.most_common_answers) 
      assert(num_answer_classes == num_answers)
    
  def train_dataloader(self):
      return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                        num_workers=self.num_workers, pin_memory=True, drop_last=True)

  def test_dataloader(self):
      return DataLoader(self.test_dataset, batch_size=self.batch_size,
                        num_workers=self.num_workers, pin_memory=True, drop_last=True)

  def get_vocab_length(self):
    return self.train_dataset.vocabulary_length()

  @staticmethod
  def is_dumb(transfer_type, num_answers, multiple_images):
    if transfer_type:
      return "_dumb_" + str(num_answers)
    else:
      if multiple_images:
        return "_multiple_"
      return ""
