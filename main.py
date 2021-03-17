from transformers import RobertaTokenizer, RobertaModel
# text = "Replace me by any text you'd like."
# encoded_input = tokenizer(text, return_tensors='pt')
# output = model(**encoded_input)
from torchvision.models import resnet18
import torch.nn as nn
import torch 
from transformers.data.data_collator import DataCollatorForLanguageModeling

class JeopardyModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base') 
        # TODO: freeze everything except the last 3 layers of Roberta
        self.model = RobertaModel.from_pretrained('roberta-base')
        self.word_embeddings = self.model.embeddings.word_embeddings
        self.image_feature_extractor = resnet18(pretrained=False)
        self.image_feature_extractor.fc = nn.Linear(512, self.word_embeddings.embedding_dim) 
        self.alpha = config.mp.alpha
        bs = config.mp.batch_size
        self.idx = torch.tensor([bs-1] + [i for i in range(1, bs-1)])
        sep_token = self.tokenizer(self.tokenizer.sep_token)
        self.sep_token = self.word_embeddings([2], return_tensors='pt')

    def forward_image(self, x):
      return self.resnet(x)

    def forward(self, x):
      out = self.model(inputs_embeds=x)
      cls_embed = out.pooler_output
      return torch.linalg.norm(cls_embed, dim=-2)

    def mlm_forward(self, x):
      # stub
      out = self.model()
      return out

    def training_step(self, batch, batch_idx):
      loss = self.shared_step(batch)
      self.log("train_loss", loss, prog_bar=True, sync_dist=True)
      return loss 


    def shared_step(self, batch):
      # recomputing the sep_token's embedding for each batch might be overkill
      
      question, image1, image2, answer = batch
      neg_image = image1.clone().view(-1)[self.idx].view(image1.size())
      image1 = self.forward_image(image1)
      image2 = self.forward_image(image2)

      # TODO: efficient thing to do might be to use cached representations from previous forward
      # passes for negatives?
      neg_image = self.forward_image(neg_image) 

      question_answer = self.tokenizer(question, answer, return_tensors='pt') # insert the SEP tokens - but not sure where padding comes in now
      
      '''
      question_answer now looks like `<s> question </s></s> answer </s>`
      </s> is the separator token; to add the image embedding in, we want
      <s> question </s></s> answer </s></s> image </s> to be the final input
      '''
      sep_token = self.sep_token
      question_answer_image1 = question_answer + sep_token + image1 + sep_token
      question_answer_image2 = question_answer + sep_token + image2 + sep_token
      question_answer_neg_image = question_answer + sep_token + neg_image + sep_token
      contrastive_loss = self(question_answer_image1) + self(question_answer_image2) - self(question_answer_neg_image)
      
      # TODO: test this util and make sure only question tokens are getting masked, 
      # default mask prob is 15% - make sure this is the behavior
      question = DataCollatorForLanguageModeling(question)
      # mlm_loss = cross_entropy_loss on labels vs predictions of masked tokens
      mlm_loss = 0
      return self.alpha*mlm_loss * (1-self.alpha)*contrastive_loss

    def configure_optimizers(self):
        # positional encodings for the image as well
        return Adam(params, lr=self.op.learning_rate, momentum=self.op.momentum, weight_decay=self.op.weight_decay)
        