from transformers import RobertaForMaskedLM, AdamW, RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaPooler
from torchvision.models import resnet18
from torch.nn import Linear
import torch 
import pytorch_lightning as pl 
from utils.model_utils import freeze_everything_except_top
import loss_func.triplet
import loss_func.simclr
import loss_func.oscar
import random 


class JeopardyModel(pl.LightningModule):
    def __init__(self, config, pretrained_bert):
        super().__init__()
        self.model = RobertaForMaskedLM.from_pretrained('roberta-base')
        freeze_everything_except_top(self.model, 3) # this seems to be too high for CCN 
        self.word_embeddings = self.model.roberta.embeddings.word_embeddings
        self.image_feature_extractor = resnet18(pretrained=False)
        self.image_feature_extractor.fc = Linear(512, self.word_embeddings.embedding_dim) 
        self.alpha = config.model_params.alpha
        self.op = config.optim_params
        self.bs = config.optim_params.batch_size
        self.q_len = config.model_params.q_len
        self.ans_len = config.model_params.ans_len
        # pretrained pooler
        self.pooler = RobertaModel.from_pretrained('roberta-base').pooler
        # self.pooler = RobertaPooler(self.model.config) if we didn't want a pretrained pooler
        self.tau = config.model_params.tau
        self.contrastive_loss = config.contrastive_loss
        self.contrastive_loss_func = loss_func.triplet.get_loss
        self.idx = torch.tensor([self.bs-1] + [i for i in range(0, self.bs-1)])
        if self.contrastive_loss == "oscar":
          self.contrastive_loss_func = loss_func.oscar.get_loss
          self.bin_classifier = Linear(768, 2)
        elif self.contrastive_loss == "simclr":
          self.contrastive_loss_func == loss_func.simclr.get_loss

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
        - sequence is always
        <start token> question <sep> <sep> answer <sep> <sep> image <end>
        (subject to change)
        TODO: understand what the consequences of the above choice are
      '''
      q, img1, img2, a = batch
      
      img1 = self.forward_image(img1)
      img2 = self.forward_image(img2)
      img1 = torch.unsqueeze(img1, dim=1)
      img2 = torch.unsqueeze(img2, dim=1)
      
      answer = self.word_embeddings(a["contrastive_input"])
      sep_token_emb = answer[:, :, -1]
      # we convert '<sos> answer <sep>' to '<sep> answer <sep>'
      answer[:, :, 0] = sep_token_emb # not memory efficient - its the exact thing 8 times
      # attention mask for (sep + img + eos)
      img_attention_mask_shape = torch.Size((self.bs, 1, 1 + 2))
      img_attention_mask = torch.ones(img_attention_mask_shape, device=self.device) # stack or cat 3 times
      attention_mask = torch.cat((q["attention_mask"], a["attention_mask"], img_attention_mask), dim=2)
      

      question = self.word_embeddings(q["contrastive_input"])
      question = question.squeeze()
      answer = answer.squeeze()
      
      '''Contrastive Loss'''
      question_answer_image1 = torch.cat((question, answer, sep_token_emb, img1, sep_token_emb), dim=1)
      question_answer_image2 = torch.cat((question, answer, sep_token_emb, img2, sep_token_emb), dim=1)
      
      
      qai1_dict = self.make_dict(question_answer_image1, attention_mask)
      qai2_dict = self.make_dict(question_answer_image2, attention_mask)

      trip1 = self(qai1_dict)
      trip2 = self(qai2_dict)

      contrastive_loss_params = {"trip1":trip1, "trip2":trip2, "tau": self.tau}
      if self.contrastive_loss == "simlcr":
        pass 
      else:
        neg_image = img1.view(img1.size())[self.idx]
        question_answer_neg_image = torch.cat((question, answer, sep_token_emb, neg_image, sep_token_emb), dim=1)
        qaineg_dict = self.make_dict(question_answer_neg_image, attention_mask)
        tripneg = self(qaineg_dict)
        contrastive_loss_params["tripneg"] = tripneg
        # if we want to get REALLY crazy, we could stack losses on top of each other
        if self.contrastive_loss == "oscar":
          labels_one = torch.ones(2*self.bs, dtype=torch.long)
          labels_zero =  torch.zeros(self.bs, dtype=torch.long)
          labels = torch.cat((labels_one, labels_zero), dim=0).to(self.device)
          combined_inputs = torch.cat((trip1, trip2, tripneg), dim=0)

          # shuffle both combined_inputs and labels in the same way
          random_seq = torch.tensor([i for i in range(3*self.bs)]) #trip1, trip2 and tripneg
          random.shuffle(random_seq)
          combined_inputs = combined_inputs.view(combined_inputs.size())[random_seq]
          labels = labels.view(labels.size())[random_seq]
          logits = self.bin_classifier(combined_inputs)
          contrastive_loss_params["labels"] = labels
          contrastive_loss_params["logits"] = logits
          
      contrastive_loss = self.contrastive_loss_func(**contrastive_loss_params)
      
      '''MLM Loss'''
      # TODO: make sure huggingface doesn't do anything to the attention masks when masking inputs
      question = self.word_embeddings(q["mlm_input"]) # has masked tokens, unlike contrastive_input
      question = question.squeeze()

      answer = self.word_embeddings(a["mlm_input"])
      answer = answer.squeeze()
      question_answer_image_mlm = torch.cat((question, answer, sep_token_emb, img1, sep_token_emb), dim=1)
      qai_mlm_dict = self.make_dict(question_answer_image_mlm, attention_mask)
      
      # loss is only calculated on masked tokens in any case
      labels = torch.ones_like(attention_mask, dtype=torch.long) * -100

      # we are setting the first (self.q_len + 2 + self.ans_len + 2) tokens to be the mlm_labels. The remaining
      # input sequence tokens correspond to the image, and are not masked
      labels[:, :, :self.q_len + 2 + self.ans_len + 2] = torch.cat((q["mlm_labels"], a["mlm_labels"]), dim=2)
      qai_mlm_dict["labels"] = labels

      mlm_loss = self.mlm_forward(qai_mlm_dict)
      return self.alpha*mlm_loss * (1-self.alpha)*contrastive_loss

    def configure_optimizers(self):
      return AdamW(self.parameters(), lr=self.op.learning_rate)

    def make_dict(self, embeds, attention_mask):
      return {
        "inputs_embeds": embeds,
        "attention_mask": attention_mask
      }