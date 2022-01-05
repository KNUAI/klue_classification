import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW

import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve, auc

import json

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--lr', type=float, default=1e-4, help='learning_rate')
parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
parser.add_argument('--gpus', type=str, default='0', help='gpu numbers')
parser.add_argument('--epochs', type=int, default=5, help='epochs')
parser.add_argument('--ckpt', type=str, default='0', help='checkpoint')
parser.add_argument('--max_length', type=int, default=256, help='max_length')
parser.add_argument('--num_labels', type=int, default=30, help='num_labels')

parser.add_argument('--relation_list_files', default='./KLUE/klue_benchmark/klue-re-v1.1/relation_list.json', type=str, help='relation_list_files_dir')
parser.add_argument('--train_files', default='./KLUE/klue_benchmark/klue-re-v1.1/klue-re-v1.1_train.json', type=str, help='train_files_dir')
parser.add_argument('--valid_files', default='./KLUE/klue_benchmark/klue-re-v1.1/klue-re-v1.1_dev.json', type=str, help='valid_files_dir')
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

def _mark_entity_spans(text: str, subject_range: (int, int), object_range: (int, int)) -> str:
    if subject_range < object_range:
        segments = [
            text[: subject_range[0]],
            "<subj>",
            text[subject_range[0] : subject_range[1] + 1],
            "</subj>",
            text[subject_range[1] + 1 : object_range[0]],
            "<obj>",
            text[object_range[0] : object_range[1] + 1],
            "</obj>",
            text[object_range[1] + 1 :],
        ]
    elif subject_range > object_range:
        segments = [
            text[: object_range[0]],
            "<obj>",
            text[object_range[0] : object_range[1] + 1],
            "</obj>",
            text[object_range[1] + 1 : subject_range[0]],
            "<subj>",
            text[subject_range[0] : subject_range[1] + 1],
            "</subj>",
            text[subject_range[1] + 1 :],
        ]
    else:
        raise ValueError("Entity boundaries overlap.")

    marked_text = "".join(segments)

    return marked_text

#only for klue
from datasets import load_dataset
input_data = load_dataset('klue', 're')
def read_data(tokenizer, paths, files):
    if files == args.train_files:
        data_list = input_data['train']
    elif files == args.valid_files:
        data_list = input_data['validation']

    sentences = []
    labels = []
    for data in data_list:
        sentence, subject_entity, object_entity, label = data['sentence'], data['subject_entity'], data['object_entity'], data['label']
        sentence = _mark_entity_spans(
            text=sentence,
            subject_range=(int(subject_entity["start_idx"]), int(subject_entity["end_idx"])),
            object_range=(int(object_entity["start_idx"]), int(object_entity["end_idx"])),
        )
        sentences.append(sentence)
        labels.append(label)

    #tokenize
    inputs = []
    segs = []
    targets = []
    for i in range(len(data_list)):
        input_dict = tokenizer(sentences[i], padding = 'max_length', truncation = True, max_length = args.max_length, return_tensors = 'pt', return_attention_mask = False)
        inputs.append(input_dict['input_ids'])
        segs.append(input_dict['token_type_ids'])
        targets.append(labels[i])

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

    sentences = []
    labels = []
    for data in data_list:
        sentence, subject_entity, object_entity, label = data['sentence'], data['subject_entity'], data['object_entity'], data['label']
        sentence = _mark_entity_spans(
            text=sentence,
            subject_range=(int(subject_entity["start_idx"]), int(subject_entity["end_idx"])),
            object_range=(int(object_entity["start_idx"]), int(object_entity["end_idx"])),
        )
        sentences.append(sentence)
        labels.append(label)
        
    #label_list
    relation_list_files_path = os.path.join(paths, args.relation_list_files)
    with open(relation_list_files_path, 'r', encoding='utf-8') as f:
        relation_class = json.load(f)['relations']
    label_list = relation_class
    label_map = {label: i for i, label in enumerate(label_list)}

    #tokenize
    inputs = []
    segs = []
    targets = []
    for i in range(len(data_list)):
        input_dict = tokenizer(sentences[i], padding = 'max_length', truncation = True, max_length = args.max_length, return_tensors = 'pt', return_attention_mask = False)
        inputs.append(input_dict['input_ids'])
        segs.append(input_dict['token_type_ids'])
        targets.append(label_map[labels[i]])

    input_tensor = torch.stack(inputs, dim=0)
    seg_tensor = torch.stack(segs, dim=0)
    mask_tensor = ~ (input_tensor == 0)

    output_tensor = torch.cat([input_tensor, seg_tensor, mask_tensor], dim=1)
    output_labels = torch.tensor(targets)

    return output_tensor, output_labels
'''

#read_data
tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
tokenizer.add_special_tokens(
    {
        "additional_special_tokens": [
            "<subj>",
            "</subj>",
            "<obj>",
            "</obj>",
        ]
    }
)
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
model.resize_token_embeddings(len(tokenizer))
model.cuda()

optimizer = AdamW(model.parameters(), lr = args.lr,  eps = 1e-8)

#model.load_state_dict(torch.load(f'./save_models/batch_{args.batch_size}_lr_{args.lr}_epochs_{args.epochs}_{args.ckpt}.pth'))

#label_list
label_list = ['no_relation', 'org:dissolved', 'org:founded', 'org:place_of_headquarters', 'org:alternate_names', 'org:member_of', 'org:members', 'org:political/religious_affiliation', 'org:product', 'org:founded_by', 'org:top_members/employees', 'org:number_of_employees/members', 'per:date_of_birth', 'per:date_of_death', 'per:place_of_birth', 'per:place_of_death', 'per:place_of_residence', 'per:origin', 'per:employee_of', 'per:schools_attended', 'per:alternate_names', 'per:parents', 'per:children', 'per:siblings', 'per:spouse', 'per:other_family', 'per:colleagues', 'per:product', 'per:religion', 'per:title']
no_relation_label_idx = label_list.index("no_relation")
label_indices = list(range(len(label_list)))
label_indices.remove(no_relation_label_idx)

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
        probs_list = []
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
            preds = torch.argmax(outputs.logits, dim=1).detach().cpu().numpy()
            probs = torch.softmax(outputs.logits, dim=1).detach().cpu().numpy()
            val_acc = np.equal(targets, preds).sum()
            val_acc_sum += val_acc
            targets_list.append(targets)
            preds_list.append(preds)
            probs_list.append(probs)

        targets_list = np.concatenate(targets_list, axis = 0)
        preds_list = np.concatenate(preds_list, axis = 0)
        probs_list = np.concatenate(probs_list, axis = 0)
        f1_scores = f1_score(targets_list, preds_list, average="macro", labels=label_indices) * 100.0

        targets_list = np.eye(30)[targets_list]
        score = np.zeros(30)
        for c in range(30):
            targets_c = targets_list.take([c], axis=1).ravel()
            preds_c = probs_list.take([c], axis=1).ravel()
            precision, recall, _ = precision_recall_curve(targets_c, preds_c)
            score[c] = auc(recall, precision)
        auprc = np.average(score) * 100.0

    print(f'  Real Validation Accuracy: {100 * val_acc_sum / len(validation_dataloader.dataset):.4f}')
    print(f'  f1_score: {f1_scores:.4f}')
    print(f'  auprc: {auprc:.4f}')






