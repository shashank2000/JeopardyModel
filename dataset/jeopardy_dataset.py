import os.path
import json
from PIL import Image
from torch.utils.data import Dataset
import torch
import os
import string
from tqdm import tqdm
from transformers import RobertaTokenizer
from collections import Counter
from transformers.data.data_collator import DataCollatorForLanguageModeling
import random
UNK_TOKEN = "<unk>"

'''
a_len_dict is 
Counter({1: 76810, 2: 20371, 3: 8048, 4: 940, 5: 231, 6: 56, 7: 7, 12: 2, 9: 2, 8: 2, 14: 1})
q_len_dict is
Counter({6: 25604, 7: 21249, 8: 18387, 5: 13061, 9: 12253, 10: 6294,
 11: 3386, 4: 2474, 12: 1902, 13: 1014,
  14: 564, 15: 324, 16: 181, 17: 113, 
  18: 77, 19: 48, 20: 34, 21: 24, 22: 6, 
  23: 2, 3: 2, 24: 2})
'''
class JeopardyDataset(Dataset):
    def __init__(self, 
        questions=[], 
        answers_file=None, 
        images_dir=None, 
        transform=None, 
        train=True, 
        q_len=8, 
        ans_len=1, 
        test_split=0.2,
        frequency_threshold=8, 
        multiple_images=False,
        mlm_probability=0.15):
        """
        Args:
            questions (string): List of the questions in random order.
            answers_file (string): Path to the json file with annotations.
            images_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            frequency_threshold: how many times does a word have to appear as an answer for it to be considered. (default: 8) 
            As used in Teney et al. https://arxiv.org/pdf/1708.02711.pdf
                
        Example call:
            test_ds = JeopardyDataset("v2_OpenEnded_mscoco_train2014_questions.json", "v2_mscoco_train2014_annotations.json", "images")

        """
        self.multiple_images = multiple_images
        self.mlm_probability = mlm_probability
        # initializing the lengths of our questions and answers
        self.q_len = q_len
        self.ans_len = ans_len
        # return test or train set?
        self.train = train

        a_f = open(answers_file, 'r')
        
        # we want only 80% of the question/answer text to build the vocabulary
        self.answers = json.load(a_f)["annotations"]
        self.questions = questions

        # don't need to worry about shuffling here because the questions are in random order
        split_index = int((1-test_split) * len(self.questions))
        
        if train:
            self.questions = self.questions[:split_index]
            self.answers = self.answers[:split_index]
        else:
            self.questions = self.questions[split_index:]
            self.answers = self.answers[split_index:]
            
        questions_dict = {q['question_id']: (q["question"], int(q["image_id"])) for q in self.questions}
        self.images_dir = images_dir

        self.image_id_to_filename = self._find_images()
        self.transform = transform    
        self.frequent_answers = self._find_frequent_answers(frequency_threshold)
        self.question_to_answer = self._find_answers()

        # we use a pretrained tokenizer when returning tokens for questions and answers
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

        # filter down questions_dict based on the answer set self.frequent_answers
        questions_dict = self.filter_questions_dict(questions_dict)
        
        self.return_array = [0] * len(questions_dict)
        
        self.i = 0
        for q in tqdm(questions_dict):
            entry = questions_dict[q]
            question_text = entry[0]
            image_id = entry[1]
            answer_set = self.question_to_answer[q]
            self.return_array[self.i] = question_text, image_id, answer_set
            self.i += 1

        self.q_len_dict = Counter()
        self.a_len_dict = Counter()
        


    def filter_questions_dict(self, question_dict):
        new_questions_dict = {}
        for q in question_dict:
            if question_dict[q][1] in self.image_id_to_filename and q in self.question_to_answer:
                new_questions_dict[q] = question_dict[q]
        return new_questions_dict

    def _find_frequent_answers(self, threshold):
        '''
            We create a set of answers across the dataset such that each member of this set has been the answer
            to a question (with confidence 'yes' or 'maybe') at least *threshold* times. 
            We pick one of these at random in __getitem__ to get diverse views.
            This is done so we are working with a smaller more refined answer set at the end of the day.  
        '''
        # TODO: more enlightened way of filtering answers. Check past VQA papers. 
        word_freq = Counter([])

        for i, ann in tqdm(enumerate(self.answers)):
            for answer in ann['answers']:
                actual_ans = answer['answer']
                if answer['answer_confidence'] != 'no':
                    word_freq[actual_ans] += 1

        return set([word for word in word_freq if word_freq[word] >= threshold])
        

    def _find_answers(self):
        """
            Each entry in self.answers is a list of answers that match to a specific
            question id. There are several answers for each question, for example, 
            here's entry 0 of a random element of self.answers:
            {"answer": "net", "answer_confidence": "maybe", "answer_id": 1}.

            For each candidate answer, we check to see if it is in self.frequent_answers,
            i.e. it has shown up *threshold* times, with confidence "yes" or "maybe" each
            time. If it is, we include it in the set of answers for the question id. 
            
            With threshold=10, there are 1683 questions with an empty answer set (none of their
            possible answers showed up in the dataset at least 10 times), and 
            353322 questions with valid answer sets

            Relevant paper: https://arxiv.org/abs/1708.02711
        """
        
        question_to_answer = {}
        num_unknowns = 0
        for i, ann in tqdm(enumerate(self.answers)):
            question_to_answer[ann["question_id"]] = set()
            for answer in ann['answers']:
                actual_ans = answer['answer']
                if actual_ans in self.frequent_answers:
                    question_to_answer[ann["question_id"]].add(actual_ans)

            if len(question_to_answer[ann["question_id"]]) == 0:
                num_unknowns += 1
                question_to_answer[ann["question_id"]].add(UNK_TOKEN)
        print("{} questions with no good answer, {} questions with valid answer sets".format(str(num_unknowns), str(len(self.answers) - num_unknowns)))
        return question_to_answer

    def _find_images(self):
        id_to_filename = {}
        for filename in os.listdir(self.images_dir):
            if not filename.endswith('.jpg'):
                continue
            id_and_extension = filename.split('_')[-1]
            image_id = int(id_and_extension.split('.')[0])
            id_to_filename[image_id] = filename
        return id_to_filename
    
    def __len__(self):
        return self.i
    
    def _process_question(self, question_text):
        # running an experiment
        # tokenized_question = self.tokenizer(
        #     question_text, return_tensors='pt')
        # contrastive_input = tokenized_question["input_ids"].clone()
        # # includes start and end
        # lengthOfInput = len(contrastive_input[0])
        # self.q_len_dict[lengthOfInput - 2] += 1

        tokenized_question = self.tokenizer(
            question_text, return_tensors='pt', padding='max_length', truncation=True, max_length=self.q_len+2)
        contrastive_input = tokenized_question["input_ids"].clone()
        collator = DataCollatorForLanguageModeling(
            self.tokenizer,
            mlm=True,
            mlm_probability=self.mlm_probability,
        )
        input_ids, labels = collator.mask_tokens(contrastive_input)
        tokenized_question['mlm_labels'] = labels
        tokenized_question['contrastive_input'] = contrastive_input
        tokenized_question['mlm_input'] = input_ids # make sure masking happens sometimes
        return tokenized_question

    def _process_answer(self, answer_text):

        # tokenized_answer = self.tokenizer(
        #     answer_text, return_tensors='pt')
        # contrastive_input = tokenized_answer["input_ids"].clone()
        # lengthOfInput = len(contrastive_input[0])
        # self.a_len_dict[lengthOfInput - 2] += 1
        # no masking right now in the answers
        tokenized_answer = self.tokenizer(
            answer_text, return_tensors='pt', padding='max_length', truncation=True, max_length=self.ans_len+2)
        contrastive_input = tokenized_answer["input_ids"].clone()
        collator = DataCollatorForLanguageModeling(
            self.tokenizer,
            mlm=True,
            mlm_probability=self.mlm_probability,
        )
        input_ids, labels = collator.mask_tokens(contrastive_input)
        tokenized_answer['mlm_labels'] = labels
        tokenized_answer['contrastive_input'] = contrastive_input
        tokenized_answer['mlm_input'] = input_ids # make sure masking happens sometimes
        return tokenized_answer

    def __getitem__(self, idx):
        # question = BatchEncoding({'attention_mask': [], 'input_ids': []}
        
        question, image_id, answer_set = self.return_array[idx]
        question = self._process_question(question)
        answer = random.choice(tuple(answer_set))
        answer = self._process_answer(answer)
        
        # we only mask question tokens, in the mlm objective, not the contrastive objective
        path = os.path.join(self.images_dir, self.image_id_to_filename[image_id])
        img = self.transform(Image.open(path).convert('RGB'))
        if self.multiple_images:
            img2 = self.transform(Image.open(path).convert('RGB'))
            return question, img, img2, answer
        return question, img, answer