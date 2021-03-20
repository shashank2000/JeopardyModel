from transformers import RobertaForMaskedLM
from transformers.models.roberta.modeling_roberta import RobertaPooler
from torchvision.models import resnet18
import torch.nn as nn
import torch 
from transformers.data.data_collator import DataCollatorForLanguageModeling
import pytorch_lightning as pl 
from utils.model_utils import freeze_everything_except_top
import random 

class JeopardyModel(pl.LightningModule):
    def __init__(self, config, pretrained_bert):
        super().__init__()
        self.model = RobertaForMaskedLM.from_pretrained('roberta-base')
        freeze_everything_except_top(self.model, 3) # this seems to be too high for CCN 
        self.word_embeddings = self.model.roberta.embeddings.word_embeddings
        self.image_feature_extractor = resnet18(pretrained=False)
        self.image_feature_extractor.fc = nn.Linear(512, self.word_embeddings.embedding_dim) 
        self.alpha = config.model_params.alpha
        self.op = config.optim_params
        bs = config.optim_params.batch_size
        self.idx = torch.tensor([bs-1] + [i for i in range(0, bs-1)])
        self.sep_token = self.word_embeddings(torch.tensor([2])) # sep token is same as end token??
        self.pooler = RobertaPooler(self.model.config) 

    def forward_image(self, x):
      return self.image_feature_extractor(x)

    def forward(self, x):
      out = self.model.roberta(**x)
      # look into the input of pooler to see what it should be
      cls_embed = self.pooler(out.last_hidden_state)
      return cls_embed

    def mlm_forward(self, x):
      out = self.model(**x)
      return out.loss

    def training_step(self, batch, batch_idx):
      loss = self.shared_step(batch)
      self.log("train_loss", loss, prog_bar=True, sync_dist=True)
      return loss 


    def shared_step(self, batch):
      '''
        some notes:
        - sep_token is the same as the eos_token in huggingface
      '''
      q, img1, img2, a = batch
      
      img1 = self.forward_image(img1)
      img2 = self.forward_image(img2)
      img1 = torch.unsqueeze(img1, dim=1)
      img2 = torch.unsqueeze(img2, dim=1)
      
      img_attention_mask = torch.ones_like(a["attention_mask"]) # sep + img + eos token
      attention_mask = torch.cat((q["attention_mask"], a["attention_mask"], img_attention_mask), dim=2)
      
      answer = self.word_embeddings(a["contrastive_input"])
      sep_token = answer[:, :, -1] 
      answer[:, :, 0] = sep_token # replace the start token in the answer embedding with the sep token
      

      question = self.word_embeddings(q["contrastive_input"])
      question = question.squeeze()
      answer = answer.squeeze()
      
      '''Contrastive Loss'''
      question_answer_image1 = torch.cat((question, answer, sep_token, img1, sep_token), dim=1)
      question_answer_image2 = torch.cat((question, answer, sep_token, img2, sep_token), dim=1)
      
      neg_image = img1.view(img1.size())[self.idx] # verify this is really shuffling stuff
      question_answer_neg_image = torch.cat((question, answer, sep_token, neg_image, sep_token), dim=1)
      
      qai1_dict = self.make_dict(question_answer_image1, attention_mask)
      qai2_dict = self.make_dict(question_answer_image2, attention_mask)
      qaineg_dict = self.make_dict(question_answer_neg_image, attention_mask)

      trip1 = self(qai1_dict)
      trip2 = self(qai2_dict)
      tripneg = self(qaineg_dict)
      
      contrastive_loss =  torch.linalg.norm(trip1.T@tripneg) - torch.linalg.norm(trip1.T@trip2) # could just use simclr class instead

      '''MLM Loss'''
      # TODO: do attention masks get modified when masking?? shouldn't right?
      question = self.word_embeddings(q["input_ids"]) # has masked tokens, unlike contrastive_input
      question = question.squeeze()

      answer = self.word_embeddings(a["input_ids"])
      answer = answer.squeeze()
      question_answer_image_mlm = torch.cat((question, answer, sep_token, img1, sep_token), dim=1)

      qai_mlm_dict = self.make_dict(question_answer_image_mlm, attention_mask)
      # loss is only calculated on masked tokens in any case
      labels = torch.zeros_like(attention_mask)
      labels[:, :, :10] = q["labels"]
      qai_mlm_dict["labels"] = labels
      mlm_loss = self.mlm_forward(qai_mlm_dict)
      # grad can only be created implicitly for scalar outputs
      return self.alpha*mlm_loss * (1-self.alpha)*contrastive_loss

    def configure_optimizers(self):
      # positional encodings for the image as well
      return torch.optim.Adam(self.parameters(), lr=self.op.learning_rate)

    def make_dict(self, embeds, attention_mask):
      return {
        "inputs_embeds": embeds,
        "attention_mask": attention_mask
      }