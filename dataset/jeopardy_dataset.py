import os.path
import json
from PIL import Image
from torch.utils.data import Dataset
import torch
import os
import string
from tqdm import tqdm
import nltk

# START_TOKEN = "<s>"
# END_TOKEN = "</s>"
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

class JeopardyDataset(Dataset):
    def __init__(self, 
        questions_file, 
        answers_file, 
        images_dir, 
        transform, 
        word2idx=None, 
        train=True, 
        q_len=8, 
        ans_len=2, 
        test_split=0.2, 
        dumb_transfer=False, 
        most_common_answers=None, 
        num_answers=0, 
        frequency_threshold=8, 
        multiple_images=False):
        """
        Args:
            questions_file (string): Path to the json file with questions.
            answers_file (string): Path to the json file with annotations.
            images_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            word2idx: word2idx[word] = index, constructed on the train set only, must be passed in for the test set.  
            frequency_threshold: how many times does a word have to appear as an answer for it to be considered. (default: 8) 
            As used in Teney et al. https://arxiv.org/pdf/1708.02711.pdf
                
        Example call:
            test_ds = JeopardyDataset("v2_OpenEnded_mscoco_train2014_questions.json", "v2_mscoco_train2014_annotations.json", "images")

        """
        self.dumb_transfer = dumb_transfer
        self.multiple_images = multiple_images
        # initializing the lengths of our questions and answers
        self.q_len = q_len
        self.ans_len = ans_len
        self.num_answers = num_answers
        # return test or train set?
        self.train = train
        self.glove_dict = self._get_glove_indices()

        q_f = open(questions_file, 'r')
        a_f = open(answers_file, 'r')
        
        # we want only 80% of the question/answer text to build the vocabulary
        self.answers = json.load(a_f)["annotations"]
        self.questions = json.load(q_f)["questions"]

        split_index = int((1-test_split) * len(self.questions))
        # we are not even using the actual test set here
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
        self.most_common_answers = most_common_answers        
        self.frequent_answers = self._find_frequent_answers(frequency_threshold) # this guy has a side effect where he builds most_common_answers in train dumb case, which isn't great
        self.question_to_answer = self._find_answers()

        # filter down questions_dict to only contain questions with images in the dataset
        questions_dict = {q: (questions_dict[q][0], questions_dict[q][1]) for q in questions_dict if questions_dict[q][1] in self.image_id_to_filename and q in self.question_to_answer}
        if train:
            # vocab made only from the train set
            self.vocab = self._make_vocab()
            self.word2idx = self._build_word2idx(self.vocab)
        else:
            self.word2idx = word2idx

        self.answer_tokens = set() # will use to build answer embedding bank in transfer task
        self.return_array = [0] * len(questions_dict) # upper bound on length
        
        self.i = 0
        for q in tqdm(questions_dict):
            entry = questions_dict[q]
            question_text = entry[0]
            image_id = entry[1]
            answer_word = self.question_to_answer[q]
            self.return_array[self.i] = question_text, image_id, answer_word 
            self.i += 1
        self.answer_tokens = list(self.answer_tokens)
        print(self.i)

    def _constrained_answers(self, a_vector_rep):
        '''
            if a_vector_rep is in the top self.num_answers - 1 most common answers (0-indexed), we map to the respective class.
            Else, we set it to self.num_answers - 1
        '''
        if a_vector_rep not in self.most_common_answers:
            return self.num_answers
        return self.most_common_answers[a_vector_rep]

        
    def _find_frequent_answers(self, threshold, only_one_word_answers=True, only_yes_confidence=False):
        '''
            We create a set of answers across the dataset such that each member of this set has been the answer
            to a question at least _threshold_ times. This is done so we are working with a smaller more refined 
            answer set at the end of the day. 
        '''
        # build a list of all the tokens, 10*num_answers in length because each answer has 10 tokens
        from collections import Counter
        word_freq = Counter([])
        print("building frequent_answers set")
        for i, ann in tqdm(enumerate(self.answers)):
            # initialized as this
            for answer in ann['answers']:
                actual_ans = answer['answer']
                if only_one_word_answers:
                    # TODO: deal with more than one word answers case
                    if " " in actual_ans:
                        # TODO: tokenization will bring the error words down, but takes too long
                        print(actual_ans)
                        continue
                    if only_yes_confidence:
                        if answer['answer_confidence'] != 'yes':
                        # needs to be among candidate answers, if not the answer is just the UNK token
                            continue
                    word_freq[actual_ans] += 1
    
        if self.dumb_transfer and self.train:
            from collections import Counter
            most_common_answers = Counter(word_freq).most_common(self.num_answers)
            mca_as_list = [k[0] for k in most_common_answers] # don't really care about the frequencies
            self.most_common_answers = {w:i for i, w in enumerate(mca_as_list)}

        return set([word for word in word_freq if word_freq[word] >= threshold])

    def _build_word2idx(self, vocab):
        '''
        word2idx[PAD_TOKEN] = len(glove_dict)
        '''
        word2idx = {w:self.glove_dict[w] for w in vocab}
        word2idx[UNK_TOKEN] = len(self.glove_dict) - 2 #sandberger is the word for unknown, we learn this embedding over time
        word2idx[PAD_TOKEN] = len(self.glove_dict) - 1
        return word2idx

    def _words_to_indices(self, sentence_array, answer=False):
        if answer:
            # will always be set to UNK_TOKEN if its more than a word long
            # what should I be doing here
            # its just a word now, "array" is misleading
            sentence_array = sentence_array.lower()
            if sentence_array not in self.word2idx:
                sentence_array = UNK_TOKEN
            return self.word2idx[sentence_array]
        return [self.word2idx[word] if word in self.word2idx else self.word2idx[UNK_TOKEN] for word in sentence_array]

    def _make_vocab(self):
        '''
        Makes the vocabulary out of the questions and corresponding answers in our 
        train set. Does not touch the test set.
        '''
        question_text = ' '.join([q['question'] for q in self.questions])
        # only include answers for questions in our train set
        answer_set = [self.question_to_answer[q['question_id']] for q in self.questions if q['question_id'] in self.question_to_answer]
        answer_text = ' '.join(answer_set)
        vocab = nltk.word_tokenize(answer_text)
        vocab.extend(nltk.word_tokenize(question_text))
        vocab = set(vocab) # takes away duplicates
        vocab = [word.lower() for word in vocab]
        copy_vocab = vocab.copy()
        for word in copy_vocab:
            # remove everything not in glove TODO: re-run Jeopardy with this
            if word not in self.glove_dict:
                vocab.remove(word)
        self.vocab_length = len(vocab)
        return vocab
        
    def _get_glove_indices(self):
        import pickle
        GLOVE_INDEX_LOC = os.environ.get('GLOVE_INDEX_LOC')
        with open(f'{GLOVE_INDEX_LOC}', 'rb') as f:
            glove_dict = pickle.load(f)
        return glove_dict

    def _preprocess_sentence(self, sentence):
        arr = nltk.word_tokenize(sentence)
        arr = [word.lower() for word in arr]
        if len(arr) > self.q_len:
            return []
        self._pad_arr(arr, self.q_len)
        # TODO: look into START_TOKEN, END_TOKEN usefulness, perhaps ablation study?
        return arr

    def _pad_arr(self, arr, length):
        while len(arr) < length:
            arr.append(PAD_TOKEN)
        
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
        # import collections
        # # loop through the questions, give each one an answer
        print("building answer mapping")
        print("size of self.frequent_answers is " + str(len(self.frequent_answers)))
        num_unknowns = 0
        for i, ann in tqdm(enumerate(self.answers)):
            # initialized as this
            question_to_answer[ann["question_id"]] = UNK_TOKEN
            for answer in ann['answers']:
                actual_ans = answer['answer']
                if actual_ans in self.frequent_answers:
                    # seems like there are certain words where the tokenizer thinks it is two words?
                    # needs to be among candidate answers, if not the answer is just the UNK token
                    question_to_answer[ann["question_id"]] = actual_ans
                    break
            if question_to_answer[ann["question_id"]] == UNK_TOKEN:
                num_unknowns += 1
        print("{} unknown answers, {} known answers".format(str(num_unknowns), str(len(self.answers) - num_unknowns)))
        return question_to_answer

    def _find_images(self):
        id_to_filename = {}
        # slightly different preprocessing in EC2
        for filename in os.listdir(self.images_dir):
            if not filename.endswith('.jpg'):
                continue
            id_and_extension = filename.split('_')[-1]
            image_id = int(id_and_extension.split('.')[0])
            id_to_filename[image_id] = filename
        return id_to_filename

    def vocabulary_length(self):
        # train has same vocab length as test set
        return self.vocab_length

    def get_split_index(self):
        '''
        Returns the start index of the test split for the dataloader to use.
        '''
        return self.test_split_start
    
    def __len__(self):
        return self.i

    def __getitem__(self, idx):
        # convert image id to image tensor on demand, not stored in RAM like that
        question, image_id, answer = self.return_array[idx]
        path = os.path.join(self.images_dir, self.image_id_to_filename[image_id])
        img = self.transform(Image.open(path).convert('RGB'))
        if self.multiple_images:
            img2 = self.transform(Image.open(path).convert('RGB'))
            return question, img, img2, answer
        return question, img, answer