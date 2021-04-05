def freeze_everything_except_top(model, num_layers):
    params = [p for p in model.parameters()]
    total_layers = len(params)
    i = 0
    while i < total_layers - num_layers:
        params[i].requires_grad = False
        i += 1

import random 
import json 

def make_shuffled_questions_file(orig_questions_file, new_questions_file):
    q_f = open(orig_questions_file, 'r')
    questions = json.load(q_f)["questions"]
    random.shuffle(questions)
    # write to new file, make it if it doesn't exist
    with open(new_questions_file, 'w') as outfile:
        json.dump(questions, outfile)
