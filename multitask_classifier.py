import time, random, numpy as np, argparse, sys, re, os
from types import SimpleNamespace
import math, random

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm

from datasets import SentenceClassificationDataset, SentencePairDataset, \
    load_multitask_data, load_multitask_test_data

from evaluation import model_eval_sst, model_eval_multitask, test_model_multitask

import proximateGD, bregmanDiv


TQDM_DISABLE=False

# fix the random seed
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5


class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        # You will want to add layers here to perform the downstream tasks.
        # Pretrain mode does not require updating bert paramters.
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True
        ### TODO

        self.sent_pair_layer = torch.nn.Sequential(
            torch.nn.Dropout(config.hidden_dropout_prob),
            torch.nn.Linear(4*BERT_HIDDEN_SIZE, BERT_HIDDEN_SIZE),
            torch.nn.LeakyReLU()
        )

        self.sentiment_classifier = torch.nn.Sequential(
            #torch.nn.Dropout(config.hidden_dropout_prob),
            #torch.nn.Linear(BERT_HIDDEN_SIZE, 32),
            #torch.nn.LeakyReLU(),
            torch.nn.Dropout(config.hidden_dropout_prob),
            #torch.nn.Linear(32, N_SENTIMENT_CLASSES)
            torch.nn.Linear(BERT_HIDDEN_SIZE, N_SENTIMENT_CLASSES)
        )

        self.paraphrase_classifier = torch.nn.Sequential(
            torch.nn.Dropout(config.hidden_dropout_prob),
            torch.nn.Linear(BERT_HIDDEN_SIZE*3, 1)
        )

        self.para_classifier = torch.nn.Sequential(
            torch.nn.Dropout(config.hidden_dropout_prob),
            torch.nn.Linear(BERT_HIDDEN_SIZE, 1)
        )

        self.similarity_classifier = torch.nn.CosineSimilarity(dim=1)

        self.sim_classifier = torch.nn.Sequential(
            torch.nn.Dropout(config.hidden_dropout_prob),
            torch.nn.Linear(BERT_HIDDEN_SIZE, 1),
            torch.nn.Sigmoid()
        )
        

    def forward(self, input_ids, attention_mask):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        ### TODO
        out_dict = self.bert(input_ids, attention_mask)

        return out_dict['pooler_output']

    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        ### TODO
        hidden = self.forward(input_ids, attention_mask)
        logits = self.sentiment_classifier(hidden)

        return logits


    def sent_pair_linear(self, input_ids_1, attention_mask_1,
                               input_ids_2, attention_mask_2, device):
        hidden_1 = self.forward(input_ids_1, attention_mask_1).to(device)
        hidden_2 = self.forward(input_ids_2, attention_mask_2).to(device)

        eps = 1e-8
        norm_h1 = (torch.norm(hidden_1, dim=1) + eps).unsqueeze(1).to(device)
        norm_h2 = (torch.norm(hidden_2, dim=1) + eps).unsqueeze(1).to(device)
        
        features = torch.cat((hidden_1, hidden_2, torch.abs(torch.sub(hidden_1, hidden_2)), 
                                (hidden_1/norm_h1) * (hidden_2/norm_h2)), dim = 1)
        
        emb = self.sent_pair_layer(features).to(device)

        return emb

    def predict_para(self, emb):
        return self.para_classifier(emb)

    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        '''
        ### TODO
        hidden_1 = self.forward(input_ids_1, attention_mask_1)
        hidden_2 = self.forward(input_ids_2, attention_mask_2)

        features = torch.cat((hidden_1, hidden_2, torch.abs(torch.sub(hidden_1, hidden_2))), dim = 1)
        logit = self.paraphrase_classifier(features)

        return logit

    def predict_sim(self, emb):
        return self.sim_classifier(emb)

    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        '''
        ### TODO
        hidden_1 = self.forward(input_ids_1, attention_mask_1)
        hidden_2 = self.forward(input_ids_2, attention_mask_2)
        logit = F.cosine_similarity(hidden_1, hidden_2)
        #logit = self.similarity_classifier(hidden_1, hidden_2)
        
        return logit

def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")


## Currently only trains on sst dataset
def train_multitask(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    
    # Load data
    # Create the data and its corresponding datasets and dataloader
    sst_train_data, num_labels,para_train_data, sts_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
    sst_dev_data, num_labels,para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='train')
    start_iteration_sst, start_iteration_sts, start_iteration_smart = 0, 0, 0

    # expand finetuning to include all multitask datasets
    if args.extension in ['rrobin', 'rrobin-smart']:
        # Args batch size is largest batch size; choose num_iterations as max len data divided by batch size
        # Update other datasets' batch size based on len data and num iterations
        if args.batch_type == "full":
            # num_iterations = min(len(sts_train_data), math.floor(len(para_train_data)/args.batch_size))
            num_iterations = math.floor(len(para_train_data)/args.batch_size)
            # batch_size_sst = math.floor(len(sst_train_data)/num_iterations)
            # batch_size_sts = math.floor(len(sts_train_data)/num_iterations)
            start_iteration_sst = num_iterations - math.floor(len(sst_train_data)/args.batch_size)
            start_iteration_sts = num_iterations - math.floor(len(sts_train_data)/args.batch_size)
            start_iteration_smart = min(start_iteration_sst, start_iteration_sts)

            sst_train_data = SentenceClassificationDataset(sst_train_data, args)
            sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size, collate_fn=sst_train_data.collate_fn)

            para_train_data = SentencePairDataset(para_train_data, args)

            sts_train_data = SentencePairDataset(sts_train_data, args, isRegression=True)
            sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size, collate_fn=sts_train_data.collate_fn)
        else:
            num_iterations = math.floor(len(sts_train_data) / args.batch_size)
            num_samples = num_iterations * args.batch_size

            sst_train_data = SentenceClassificationDataset(random.sample(sst_train_data, num_samples), args)
            sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size, collate_fn=sst_train_data.collate_fn)

            para_train_data = SentencePairDataset(random.sample(para_train_data, num_samples), args)
            
            sts_train_data = SentencePairDataset(random.sample(sts_train_data, num_samples), args, isRegression=True)
            sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size, collate_fn=sts_train_data.collate_fn)

        sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)
        sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size, collate_fn=sst_dev_data.collate_fn)

        para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size, collate_fn=para_train_data.collate_fn)

        para_dev_data = SentencePairDataset(para_dev_data, args)
        para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size, collate_fn=para_dev_data.collate_fn)

        sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)
        sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size, collate_fn=sts_dev_data.collate_fn)

    else:  # default train only on sentiment dataset
        sst_train_data = SentenceClassificationDataset(sst_train_data, args)
        sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

        sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size, collate_fn=sst_train_data.collate_fn)
        sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size, collate_fn=sst_dev_data.collate_fn)

    # Init model
    config = SimpleNamespace(
        hidden_dropout_prob = args.hidden_dropout_prob,
        num_labels = num_labels,
        hidden_size = 768,
        data_dir = '.',
        option = args.option,
        extension=args.extension,
        pgd_k=args.pgd_k,
        pgd_epsilon=args.pgd_epsilon,
        pgd_lambda=args.pgd_lambda,
        mbpp_beta=args.mbpp_beta,
        mbpp_mu=args.mbpp_mu
    )

    model = MultitaskBERT(config)
    model = model.to(device)

    if args.extension in ['smart', 'rrobin-smart']:
        pgd = proximateGD.AdversarialReg(model, args.pgd_epsilon, args.pgd_lambda)
        mbpp = bregmanDiv.MBPP(model, args.mbpp_beta, args.mbpp_mu)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    best_dev_acc = 0

    # expand finetuning to include all multitask datasets
    if args.extension in ['rrobin', 'rrobin-smart']:

        # run for the specified number of epochs
        for epoch in range(args.epochs):
            model.train()
            num_batches = 0

            train_loss_sst, train_loss_para, train_loss_sts = 0, 0, 0

            sst_iterator = iter(sst_train_dataloader)
            para_iterator = iter(para_train_dataloader)
            sts_iterator = iter(sts_train_dataloader)

            for _ in tqdm(range(num_iterations), desc=f'train-{epoch}', disable=TQDM_DISABLE):
                if num_batches >= start_iteration_sst:  #TODO: can only execute if statement when batch_type=full
                    ### sentiment ----------------------------------------------
                    batch = next(sst_iterator)
                    b_ids, b_mask, b_labels = (batch['token_ids'], batch['attention_mask'], batch['labels'])
                    b_ids, b_mask, b_labels = b_ids.to(device), b_mask.to(device), b_labels.to(device)

                    optimizer.zero_grad()
                    logits = model.predict_sentiment(b_ids, b_mask)
                    loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size

                    loss.backward(retain_graph=True)  # added retain_graph=True

                    if args.extension in ['rrobin-smart', 'smart']:  # smart regularization
                        # adversarial loss
                        batch_inputs = (b_ids, b_mask)
                        adv_loss = pgd.max_loss_reg(batch_inputs, logits, task_name='sst')
                        adv_loss.backward(retain_graph=True)

                        # bregman divergence
                        breg_div = mbpp.bregman_divergence(batch_inputs, logits, task_name='sst')
                        breg_div.backward(retain_graph=True)

                        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
                        optimizer.step()
                        mbpp.apply_momentum(model.named_parameters())

                    else:  # default implementation
                        #nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
                        optimizer.step()

                    train_loss_sst += loss.item()

                ### paraphrase ----------------------------------------------
                batch = next(para_iterator)
                (b_ids1, b_mask1, b_ids2, b_mask2, b_labels) = (batch['token_ids_1'], batch['attention_mask_1'], batch['token_ids_2'], batch['attention_mask_2'], batch['labels'])
                b_ids1, b_mask1, b_ids2, b_mask2, b_labels = b_ids1.to(device), b_mask1.to(device), b_ids2.to(device), b_mask2.to(device), b_labels.to(device)

                optimizer.zero_grad()
                logits = model.predict_paraphrase(b_ids1, b_mask1, b_ids2, b_mask2)
                #logits = model.predict_para(model.sent_pair_linear(b_ids1, b_mask1, b_ids2, b_mask2, device))
                loss = F.binary_cross_entropy(logits.squeeze().sigmoid(), b_labels.view(-1).type(torch.float32), reduction='sum') / args.batch_size

                loss.backward(retain_graph=True)  # added retain_graph=True

                if args.extension in ['rrobin-smart', 'smart'] and num_batches >= start_iteration_smart:  # smart regularization

                    # adversarial loss
                    batch_inputs = (b_ids1, b_mask1, b_ids2, b_mask2)
                    adv_loss = pgd.max_loss_reg(batch_inputs, logits, task_name='para')
                    adv_loss.backward(retain_graph=True)

                    # bregman divergence
                    breg_div = mbpp.bregman_divergence(batch_inputs, logits, task_name='para')
                    breg_div.backward(retain_graph=True)

                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
                    optimizer.step()
                    mbpp.apply_momentum(model.named_parameters())

                else:  # default implementation
                    #nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
                    optimizer.step()

                train_loss_para += loss.item()

                if num_batches >= start_iteration_sts:
                    ### similarity ----------------------------------------------
                    batch = next(sts_iterator)
                    (b_ids1, b_mask1, b_ids2, b_mask2, b_labels) = (batch['token_ids_1'], batch['attention_mask_1'], batch['token_ids_2'], batch['attention_mask_2'], batch['labels'])
                    b_ids1, b_mask1, b_ids2, b_mask2, b_labels = b_ids1.to(device), b_mask1.to(device), b_ids2.to(device), b_mask2.to(device), b_labels.to(device)

                    optimizer.zero_grad()
                    logits = model.predict_similarity(b_ids1, b_mask1, b_ids2, b_mask2)
                    #logits = model.predict_sim(model.sent_pair_linear(b_ids1, b_mask1, b_ids2, b_mask2, device))
                    b_labels_scaled = (b_labels/5.0).type(torch.float32)
                    loss = F.mse_loss(logits.flatten(), b_labels_scaled.view(-1))

                    loss.backward(retain_graph=True)  # added retain_graph=True

                    if args.extension in ['rrobin-smart', 'smart']:  # smart regularization

                        # adversarial loss
                        batch_inputs = (b_ids1, b_mask1, b_ids2, b_mask2)
                        adv_loss = pgd.max_loss_reg(batch_inputs, logits, task_name='sts')
                        adv_loss.backward(retain_graph=True)

                        # bregman divergence 
                        breg_div = mbpp.bregman_divergence(batch_inputs, logits, task_name='sts')
                        breg_div.backward(retain_graph=True)

                        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
                        optimizer.step()
                        mbpp.apply_momentum(model.named_parameters())

                    else:  # default implementation
                        #nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
                        optimizer.step()

                    train_loss_sts += loss.item()

                num_batches += 1

            train_loss_sst = train_loss_sst / (num_batches)
            train_loss_para = train_loss_para / (num_batches)
            train_loss_sts = train_loss_sts / (num_batches)

            (train_acc_para, _, _,
            train_acc_sst, _, _,
            train_acc_sts, _, _) = model_eval_multitask(sst_train_dataloader, 
                                                        para_train_dataloader, 
                                                        sts_train_dataloader, 
                                                        model,
                                                        device,
                                                        False)

            (dev_acc_para, _, _,
            dev_acc_sst, _, _,
            dev_acc_sts, _, _) = model_eval_multitask(sst_dev_dataloader, 
                                                    para_dev_dataloader, 
                                                    sts_dev_dataloader, 
                                                    model,
                                                    device,
                                                    False)

            dev_acc = np.mean([dev_acc_sst, dev_acc_para, dev_acc_sts])
            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                save_model(model, optimizer, args, config, args.filepath)

            print(f"Epoch {epoch}: sentiment -->> train loss :: {train_loss_sst :.3f}, train acc :: {train_acc_sst :.3f}, dev acc :: {dev_acc_sst :.3f}") 
            print(f"Epoch {epoch}: paraphrase -->> train loss :: {train_loss_para :.3f}, train acc :: {train_acc_para :.3f}, dev acc :: {dev_acc_para :.3f}") 
            print(f"Epoch {epoch}: similarity -->> train loss :: {train_loss_sts :.3f}, train acc :: {train_acc_sts :.3f}, dev acc :: {dev_acc_sts :.3f}") 

            # del loss; del adv_loss; del breg_div; del logits

    else:  # default train only on sentiment dataset

        # run for the specified number of epochs
        for epoch in range(args.epochs):
            model.train()
            num_batches = 0
            train_loss = 0

            for batch in tqdm(sst_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
                b_ids, b_mask, b_labels = (batch['token_ids'], batch['attention_mask'], batch['labels'])
                b_ids, b_mask, b_labels = b_ids.to(device), b_mask.to(device), b_labels.to(device)

                optimizer.zero_grad()
                logits = model.predict_sentiment(b_ids, b_mask)
                loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size

                loss.backward(retain_graph=True)  # added retain_graph=True

                # smart regularization
                if args.extension in ['rrobin-smart', 'smart']:

                    # adversarial loss
                    batch_inputs = (b_ids, b_mask)
                    adv_loss = pgd.max_loss_reg(batch_inputs, logits, task_name='sst')
                    adv_loss.backward(retain_graph=True)

                    # bregman divergence
                    breg_div = mbpp.bregman_divergence(batch_inputs, logits, task_name='sst')
                    breg_div.backward(retain_graph=True)

                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
                    optimizer.step()
                    mbpp.apply_momentum(model.named_parameters())

                else:  # default implementation
                    optimizer.step()

                train_loss += loss.item()
                num_batches += 1

            train_loss = train_loss / (num_batches)
                
            train_acc, train_f1, *_ = model_eval_sst(sst_train_dataloader, model, device)
            dev_acc, dev_f1, *_ = model_eval_sst(sst_dev_dataloader, model, device)

            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                save_model(model, optimizer, args, config, args.filepath)
            
            print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}") 

            # del loss; del logits

def test_model(args):
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        test_model_multitask(args, model, device)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune'), default="pretrain")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    # hyper parameters
    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--batch_type", type=str, default="small")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-5)
    parser.add_argument("--extension", type=str, default="default")

    # adversarial regularization
    parser.add_argument('--pgd_k', type=int, default=1)
    parser.add_argument('--pgd_epsilon', type=float, default=1e-5)
    parser.add_argument('--pgd_lambda', type=float, default=10)

    # bregman momentum
    parser.add_argument('--mbpp_beta', type=float, default=0.995)
    parser.add_argument('--mbpp_mu', type=float, default=1)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.option}-{args.epochs}-{args.lr}-multitask.pt' # save path
    seed_everything(args.seed)  # fix the seed for reproducibility
    train_multitask(args)
    test_model(args)
