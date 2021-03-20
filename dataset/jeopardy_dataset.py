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

UNK_TOKEN = "<unk>"

# TODO: no restriction on answers
class JeopardyDataset(Dataset):
    def __init__(self, 
        questions_file=None, 
        answers_file=None, 
        images_dir=None, 
        transform=None, 
        train=True, 
        q_len=8, 
        ans_len=1, 
        test_split=0.2,
        frequency_threshold=8, 
        multiple_images=False):
        """
        Args:
            questions_file (string): Path to the json file with questions.
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
        # initializing the lengths of our questions and answers
        self.q_len = q_len
        self.ans_len = ans_len
        # return test or train set?
        self.train = train

        q_f = open(questions_file, 'r')
        a_f = open(answers_file, 'r')
        
        # we want only 80% of the question/answer text to build the vocabulary
        self.answers = json.load(a_f)["annotations"]
        self.questions = json.load(q_f)

        # don't need to worry about shuffling here because the questions
        # in the file are in random order (ctsy utils.model_utils:shuffled_questions_file()) 
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
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

        # filter down questions_dict based on the answer set self.frequent_answers
        questions_dict = self.filter_questions_dict(questions_dict)
        
        self.return_array = [0] * len(questions_dict)
        
        self.i = 0
        for q in tqdm(questions_dict):
            entry = questions_dict[q]
            question_text = entry[0]
            image_id = entry[1]
            answer_text = self.question_to_answer[q]
            self.return_array[self.i] = question_text, image_id, answer_text
            self.i += 1
        


    def filter_questions_dict(self, question_dict):
        new_questions_dict = {}
        for q in question_dict:
            if question_dict[q][1] in self.image_id_to_filename and q in self.question_to_answer:
                new_questions_dict[q] = question_dict[q]
        return new_questions_dict

    def _find_frequent_answers(self, threshold, only_one_word_answers=True, only_yes_confidence=False):
        '''
            We create a set of answers across the dataset such that each member of this set has been the answer
            to a question at least *threshold* times. This is done so we are working with a smaller more refined 
            answer set at the end of the day. We also filter the set of answers so we are dealing with only 
            answer_confidence deals with the subject's confidence in answering the question, we are currently filtering
            so its only yes's considered. I believe no and maybe are two other categories. 
        '''
        # TODO: more enlightened way of filtering answers. Check past VQA papers. 
        word_freq = Counter([])
        for i, ann in tqdm(enumerate(self.answers)):
            for answer in ann['answers']:
                actual_ans = answer['answer']
                if only_one_word_answers and " " in actual_ans:
                        continue
                if only_yes_confidence:
                    if answer['answer_confidence'] != 'yes':
                        continue
                word_freq[actual_ans] += 1
    
        return set([word for word in word_freq if word_freq[word] >= threshold])
        

    def _find_answers(self):
        """
            There several answers for each question in the dataset, for example, 
            here's entry 0 of the answer vector for a random question:
            {"answer": "net", "answer_confidence": "maybe", "answer_id": 1}
            
            For the sake of simplicity, we only consider answers that have answer_confidence: yes,
            and are exactly one word long. We will apply a similar padding paradigm as we do with the question
            texts later on, by using the _pad_arr function. One to one correspondence between answers and questions, 
            as in there is one answers object for each question, and so its fair to go only until the test split for both.
        """
        
        question_to_answer = {}
        num_unknowns = 0
        for i, ann in tqdm(enumerate(self.answers)):
            question_to_answer[ann["question_id"]] = UNK_TOKEN
            for answer in ann['answers']:
                actual_ans = answer['answer']
                if actual_ans in self.frequent_answers:
                    question_to_answer[ann["question_id"]] = actual_ans
                    break
            if question_to_answer[ann["question_id"]] == UNK_TOKEN:
                num_unknowns += 1
        print("{} unknown answers, {} known answers".format(str(num_unknowns), str(len(self.answers) - num_unknowns)))
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
        tokenized_question = self.tokenizer(
            question_text, return_tensors='pt', padding='max_length', truncation=True, max_length=self.q_len+2)
        contrastive_input = tokenized_question["input_ids"].clone()
        collator = DataCollatorForLanguageModeling(
            self.tokenizer,
            mlm=True,
            mlm_probability=0.15,
        )
        input_ids, labels = collator.mask_tokens(contrastive_input)
        tokenized_question['labels'] = labels
        tokenized_question['contrastive_input'] = contrastive_input
        tokenized_question['input_ids'] = input_ids # make sure masking happens sometimes
        return tokenized_question

    def _process_answer(self, answer_text):
        # no masking right now in the answers
        tokenized_answer = self.tokenizer(
            answer_text, return_tensors='pt', padding='max_length', truncation=True, max_length=self.ans_len+2)
        contrastive_input = tokenized_answer["input_ids"].clone()
        collator = DataCollatorForLanguageModeling(
            self.tokenizer,
            mlm=True,
            mlm_probability=0.15,
        )
        input_ids, labels = collator.mask_tokens(contrastive_input)
        tokenized_answer['labels'] = labels
        tokenized_answer['contrastive_input'] = contrastive_input
        tokenized_answer['input_ids'] = input_ids # make sure masking happens sometimes
        return tokenized_answer

    def __getitem__(self, idx):
        # question = BatchEncoding({'attention_mask': [], 'input_ids': []}
        question, image_id, answer = self.return_array[idx]
        question = self._process_question(question)
        answer = self._process_answer(answer)
        # we only mask question tokens, in the mlm objective, not the contrastive objective
        path = os.path.join(self.images_dir, self.image_id_to_filename[image_id])
        img = self.transform(Image.open(path).convert('RGB'))
        if self.multiple_images:
            img2 = self.transform(Image.open(path).convert('RGB'))
            return question, img, img2, answer
        return question, img, answer