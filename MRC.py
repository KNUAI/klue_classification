# -*- coding: cp949 -*-
# torch==1.4.0
# transformers==3.5.1
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AdamW
from transformers.data.processors.squad import SquadProcessor, squad_convert_examples_to_features, SquadExample, SquadResult
from transformers.data.metrics.squad_metrics import compute_predictions_logits, squad_evaluate

import numpy as np
from sklearn.metrics import f1_score

import json

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--lr', type=float, default=3e-5, help='learning_rate')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
parser.add_argument('--gpus', type=str, default='0', help='gpu numbers')
parser.add_argument('--epochs', type=int, default=5, help='epochs')
parser.add_argument('--ckpt', type=str, default='0', help='checkpoint')
parser.add_argument('--max_seq_length', type=int, default=448, help='max_seq_length')
parser.add_argument('--max_query_length', type=int, default=64, help='max_query_length')
parser.add_argument('--num_labels', type=int, default=2, help='num_labels')

parser.add_argument('--train_files', default='./KLUE/klue_benchmark/klue-mrc-v1.1/klue-mrc-v1.1_train.json', type=str, help='train_files_dir')
parser.add_argument('--valid_files', default='./KLUE/klue_benchmark/klue-mrc-v1.1/klue-mrc-v1.1_dev.json', type=str, help='valid_files_dir')
parser.add_argument('--test_files', default='./', type=str, help='test_files_dir')

parser.add_argument('--pretrained_model', default='klue/bert-base', type=str, help='pretrained_model_name')
parser.add_argument('--seed', type=int, default=1234, help='seed')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#seed
if args.seed is not None:
    import random
    import numpy as np
    import torch
    import torch.backends.cudnn as cudnn
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = True

class SquadV1Processor(SquadProcessor):    
    train_file = args.train_files
    dev_file = args.valid_files
    
    def _create_examples(self, input_data, set_type):
        is_training = set_type == "train"
        examples = []
        for entry in input_data:
            title = entry["title"]
            for paragraph in entry["paragraphs"]:
                context_text = paragraph["context"]
                for qa in paragraph["qas"]:
                    qas_id = qa["guid"]
                    question_text = qa["question"]
                    start_position_character = None
                    answer_text = None
                    answers = []

                    is_impossible = qa.get("is_impossible", False)
                    if not is_impossible:
                        if is_training:
                            answer = qa["answers"][0]
                            answer_text = answer["text"]
                            start_position_character = answer["answer_start"]
                        else:
                            answers = qa["answers"]

                    example = SquadExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        context_text=context_text,
                        answer_text=answer_text,
                        start_position_character=start_position_character,
                        title=title,
                        is_impossible=is_impossible,
                        answers=answers,
                    )
                    examples.append(example)
        return examples    

def read_data(tokenizer, mode):
    paths = os.getcwd()
    processor = SquadV1Processor()    
    if mode == 'train':
        examples = processor.get_train_examples(paths, filename = processor.train_file)
    elif mode == 'dev':
        examples = processor.get_dev_examples(paths, filename=processor.dev_file)

    features, dataset = squad_convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        doc_stride=128,
        max_query_length=args.max_query_length,
        is_training=True if mode == 'train' else False,
        return_dataset='pt',
        threads=4,
    )

    return dataset, examples, features

#read_data
tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
train_data, _, _ = read_data(tokenizer, 'train')
validation_data, examples, features = read_data(tokenizer, 'dev')
#test_data, test_examples, test_features = read_data(tokenizer, 'test')

#data_loader
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)

validation_dataloader = DataLoader(validation_data, sampler=None, batch_size=args.batch_size)
#test_dataloader = DataLoader(test_data, sampler=None, batch_size=args.batch_size)

model = AutoModelForQuestionAnswering.from_pretrained(args.pretrained_model, num_labels=args.num_labels)
model.cuda()

optimizer = AdamW(model.parameters(), lr = args.lr,  eps = 1e-8)

#model.load_state_dict(torch.load(f'./save_models/batch_{args.batch_size}_lr_{args.lr}_epochs_{args.epochs}_{args.ckpt}.pth'))

for epoch_i in range(args.epochs):

    #train
    model.train()
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, args.epochs))
    print('Training...')

    total_loss = 0
    train_loss = []
    for step, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        if step % 100 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))

        inputs = {
            "input_ids": batch[0].to(device),
            "attention_mask": batch[1].to(device),
            "token_type_ids": batch[2].to(device),
            "start_positions": batch[3].to(device),
            "end_positions": batch[4].to(device)
        }

        outputs = model(**inputs)                        
        loss = outputs[0]
        
        total_loss += loss.item()
        train_loss.append(total_loss/(step+1))

        loss.backward()
        optimizer.step()

    avg_train_loss = total_loss / len(train_dataloader)

    print(f'  Average training loss: {avg_train_loss:.2f}')
    '''
    if not os.path.exists('./save_models'):
        os.makedirs('./save_models')
    '''
    #torch.save(model.state_dict(), f'./save_models/batch_{args.batch_size}_lr_{args.lr}_epochs_{args.epochs}_{args.ckpt}.pth')

    #validation
    with torch.no_grad():
        model.eval()
        print('Running Real Validation...')

        all_results = []
        for batch in validation_dataloader:

            inputs = {
                "input_ids": batch[0].to(device),
                "attention_mask": batch[1].to(device),
                "token_type_ids": batch[2].to(device)
            }

            feature_indices = batch[3].to(device)
            outputs = model(**inputs)
            
            for i, feature_index in enumerate(feature_indices):
                eval_feature = features[feature_index.item()]
                unique_id = int(eval_feature.unique_id)

                start_logits = outputs[0][i]
                end_logits = outputs[1][i]                
                result = SquadResult(unique_id, start_logits, end_logits)
    
                all_results.append(result)
        predictions = compute_predictions_logits(examples,        
                                                features,
                                                all_results,
                                                n_best_size=20,
                                                max_answer_length=30,
                                                do_lower_case=True,
                                                output_prediction_file=None,
                                                output_nbest_file=None,
                                                output_null_log_odds_file=None,
                                                verbose_logging=False,
                                                version_2_with_negative=False,
                                                null_score_diff_threshold=0.0,
                                                tokenizer=tokenizer)
        results = squad_evaluate(examples, predictions)

        print(f"  em_score: {results['exact']:.4f}")
        print(f"  rouge_w_score: {results['f1']:.4f}")






