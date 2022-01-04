import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW

import numpy as np
from sklearn.metrics import f1_score
from scipy.stats import pearsonr

import json

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--lr', type=float, default=1e-4, help='learning_rate')
parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
parser.add_argument('--gpus', type=str, default='0', help='gpu numbers')
parser.add_argument('--epochs', type=int, default=5, help='epochs')
parser.add_argument('--ckpt', type=str, default='0', help='checkpoint')
parser.add_argument('--max_length', type=int, default=128, help='max_length')
parser.add_argument('--num_labels', type=int, default=1, help='num_labels')

parser.add_argument('--train_files', default='./KLUE/klue_benchmark/klue-sts-v1.1/klue-sts-v1.1_train.json', type=str, help='train_files_dir')
parser.add_argument('--valid_files', default='./KLUE/klue_benchmark/klue-sts-v1.1/klue-sts-v1.1_dev.json', type=str, help='valid_files_dir')
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

#only for klue
from datasets import load_dataset
input_data = load_dataset('klue', 'sts')
def read_data(tokenizer, paths, files):
    if files == args.train_files:
        data_list = input_data['train']
    elif files == args.valid_files:
        data_list = input_data['validation']

    sentence1s = []
    sentence2s = []
    real_labels = []
    for data in data_list:
        sentence1, sentence2, real_label = data['sentence1'], data['sentence2'], data['labels']['real-label']
        sentence1s.append(sentence1)
        sentence2s.append(sentence2)
        real_labels.append(real_label)

    #tokenize
    inputs = []
    segs = []
    targets = []
    for i in range(len(data_list)):
        input_dict = tokenizer(sentence1s[i], sentence2s[i], padding = 'max_length', truncation = True, max_length = args.max_length, return_tensors = 'pt', return_attention_mask = False)
        inputs.append(input_dict['input_ids'])
        segs.append(input_dict['token_type_ids'])
        targets.append(real_labels[i])

    input_tensor = torch.stack(inputs, dim=0)
    seg_tensor = torch.stack(segs, dim=0)
    mask_tensor = ~ (input_tensor == 0)

    output_tensor = torch.cat([input_tensor, seg_tensor, mask_tensor], dim=1)
    output_labels = torch.tensor(targets)

    return output_tensor, output_labels

'''
def read_data(tokenizer, paths, files):
    #load_data
    file_path = os.path.join(paths, files)
    with open(file_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    sentence1s = []
    sentence2s = []
    real_labels = []
    for data in data_list:
        sentence1, sentence2, real_label = data['sentence1'], data['sentence2'], data['labels']['real-label']
        sentence1s.append(sentence1)
        sentence2s.append(sentence2)
        real_labels.append(real_label)

    #tokenize
    inputs = []
    segs = []
    targets = []
    for i in range(len(data_list)):
        input_dict = tokenizer(sentence1s[i], sentence2s[i], padding = 'max_length', max_length = args.max_length, return_tensors = 'pt', return_attention_mask = False)
        inputs.append(input_dict['input_ids'])
        segs.append(input_dict['token_type_ids'])
        targets.append(real_labels[i])

    input_tensor = torch.stack(inputs, dim=0)
    seg_tensor = torch.stack(segs, dim=0)
    mask_tensor = ~ (input_tensor == 0)

    output_tensor = torch.cat([input_tensor, seg_tensor, mask_tensor], dim=1)
    output_labels = torch.tensor(targets)

    return output_tensor, output_labels
'''

#read_data
tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
paths = os.getcwd()
train_inputs, train_labels = read_data(tokenizer, paths, args.train_files)
valid_inputs, valid_labels = read_data(tokenizer, paths, args.valid_files)
#test_inputs, _ = read_data(tokenizer, paths, args.test_files)

#data_loader
train_data = TensorDataset(train_inputs, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)

validation_data = TensorDataset(valid_inputs, valid_labels)
validation_dataloader = DataLoader(validation_data, sampler=None, batch_size=args.batch_size)

#test_data = TensorDataset(test_inputs, test_inputs)
#test_dataloader = DataLoader(test_data, sampler=None, batch_size=args.batch_size)

model = AutoModelForSequenceClassification.from_pretrained(args.pretrained_model, num_labels=args.num_labels)
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
    for step, (data, labels) in enumerate(train_dataloader):
        optimizer.zero_grad()
        if step % 100 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))

        src = data[:, 0, :]
        segs = data[:, 1, :]
        mask = data[:, 2, :]

        src = src.to(device)
        segs = segs.to(device)
        mask = mask.to(device)
        labels = labels.to(device)

        outputs = model(src, 
                        token_type_ids=segs, 
                        attention_mask=mask, 
                        labels=labels)

        loss = outputs.loss
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

        val_acc_sum = 0
        targets_list = []
        preds_list = []
        for data, labels in validation_dataloader:

            src = data[:, 0, :]
            segs = data[:, 1, :]
            mask = data[:, 2, :]

            src = src.to(device)
            segs = segs.to(device)
            mask = mask.to(device)
            labels = labels.to(device)

            outputs = model(src,
                            token_type_ids=segs,
                            attention_mask=mask)

            targets = labels.detach().cpu().numpy()
            preds = outputs.logits.detach().cpu().numpy()
            preds = np.squeeze(preds)
            targets_list.append(targets)
            preds_list.append(preds)


        targets_list = np.concatenate(targets_list, axis = 0)
        preds_list = np.concatenate(preds_list, axis = 0)
        pearsonrs = pearsonr(preds_list, targets_list)[0] * 100.0
        
        preds_list = np.where(preds_list >= 3, 1, 0)
        targets_list = np.where(targets_list >= 3, 1, 0)
        f1_scores = f1_score(targets_list, preds_list, average="macro") * 100.0

    print(f'  pearsonr: {pearsonrs:.4f}')
    print(f'  f1_score: {f1_scores:.4f}')






