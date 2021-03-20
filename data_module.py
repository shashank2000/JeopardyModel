from torchvision import transforms
from dataset.jeopardy_dataset import JeopardyDataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from torch import tensor
from PIL import ImageFilter
from utils.model_utils import make_shuffled_questions_file
import random
import torch
import os
import json 

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""
    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = sigma
    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
        
class VQADataModule(LightningDataModule):
  def __init__(self, batch_size, threshold=10, q_len=8, num_workers=8, val_split=0.2, num_answers=0, transfer=False, multiple_images=False):
    super().__init__()

    self.questions_file = self._make_shuffled_questions_file()

    self.answers_file = os.environ.get('ANSWERS_FILE')
    self.coco_loc = os.environ.get('COCO_LOC')   

    self.batch_size = batch_size

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
    
    saved_train_file = 'train{}.pt'.format(self.file_name(multiple_images))
    saved_test_file = 'test{}.pt'.format(self.file_name(multiple_images))

    try:
        print("attempting to use cache")
        # check if git diff jeopardy_dataset.py is empty, if so continue, else make new caches
        self.train_dataset = torch.load(saved_train_file)
        self.test_dataset = torch.load(saved_test_file)
        print("found files")
    except:
        print(saved_train_file + " not found")

        # threshold will be important when doing finetuning
        self.train_dataset = JeopardyDataset(
          questions_file=self.questions_file, 
          answers_file=self.answers_file, 
          images_dir=self.coco_loc, 
          transform=self.train_transform,
          frequency_threshold=threshold, 
          train=True,
          multiple_images=multiple_images)

        self.test_dataset = JeopardyDataset(
          questions_file=self.questions_file, 
          answers_file=self.answers_file, 
          images_dir=self.coco_loc, 
          transform=self.train_transform,
          frequency_threshold=threshold, 
          train=False,
          multiple_images=multiple_images)
        
        torch.save(self.train_dataset, saved_train_file)
        torch.save(self.test_dataset, saved_test_file)
    self.num_workers = num_workers
    
  def train_dataloader(self):
      return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                        num_workers=self.num_workers, pin_memory=True, drop_last=True)

  @staticmethod
  def file_name(multiple_images):
    # cache file name
    if multiple_images:
      return "multiple"
    return ""

  def _make_shuffled_questions_file(self):
    # if there is no shuffled questions file, make one
    shuffled_questions_file = os.environ.get('SHUFFLED_QUESTIONS_FILE')
    if not os.path.isfile(shuffled_questions_file):
      original_questions_file = os.environ.get('QUESTIONS_FILE')
      make_shuffled_questions_file(original_questions_file, shuffled_questions_file)
    return shuffled_questions_file
