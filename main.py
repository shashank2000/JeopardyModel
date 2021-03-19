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
        # TODO: freeze everything except the last 3 layers of Roberta
        self.model = RobertaForMaskedLM.from_pretrained('roberta-base')
        freeze_everything_except_top(self.model, 3)
        self.word_embeddings = self.model.roberta.embeddings.word_embeddings
        self.image_feature_extractor = resnet18(pretrained=False)
        self.image_feature_extractor.fc = nn.Linear(512, self.word_embeddings.embedding_dim) 
        self.alpha = config.model_params.alpha
        self.op = config.optim_params
        bs = config.optim_params.batch_size
        self.idx = torch.tensor([bs-1] + [i for i in range(0, bs-1)])
        self.sep_token = self.word_embeddings(torch.tensor([2]))
        self.pooler = RobertaPooler(self.model.config)

    def forward_image(self, x):
      return self.image_feature_extractor(x)

    def forward(self, x):
      out = self.model.roberta(inputs_embeds=x, return_dict=True)
      cls_embed = self.pooler(out)
      return torch.linalg.norm(cls_embed, dim=-2)

    def mlm_forward(self, x, labels):
      out = self.model(inputs_embeds=x, labels=labels, return_dict=True)
      return out.loss

    def training_step(self, batch, batch_idx):
      loss = self.shared_step(batch)
      self.log("train_loss", loss, prog_bar=True, sync_dist=True)
      return loss 


    def shared_step(self, batch):
      # recomputing the sep_token's embedding for each batch might be overkill
      
      question, image1, image2, answer = batch
      # they are tokens, this is where we can mask a question token
      q_labels = torch.ones_like(question['input_ids']) * -100
      random_word = random.randint(0, len(question))
      q_labels[random_word] = question[random_word]
      breakpoint()
      image1 = self.forward_image(image1)
      image2 = self.forward_image(image2)
      question_answer = self.tokenizer(question, answer, return_tensors='pt', padding=True) 

      '''
      question_answer now looks like `<s> question </s></s> answer </s>`
      </s> is the separator token; to add the image embedding in, we want
      <s> question </s></s> answer </s></s> image </s> to be the final input
      '''
      sep_token = self.sep_token
      breakpoint()
      question_answer_image1 = torch.cat(question_answer, sep_token, image1, sep_token)
      question_answer_image2 = torch.cat(question_answer, sep_token, image2, sep_token)
      
      neg_image = image1.view(image1.size())[self.idx]
      question_answer_neg_image = question_answer + sep_token + neg_image + sep_token
      contrastive_loss = self(question_answer_image1) + self(question_answer_image2) - self(question_answer_neg_image)
      
      mlm_loss = self.mlm_forward(question_answer_image1, labels=q_labels)
      return self.alpha*mlm_loss * (1-self.alpha)*contrastive_loss

    def configure_optimizers(self):
        # positional encodings for the image as well
        return torch.optim.Adam(self.parameters(), lr=self.op.learning_rate)
        