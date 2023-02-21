import torch

from bert import BertModel
from typing import Dict, List, Optional, Union, Tuple, Callable




class proximateGD(object):
    def __init__(self,
            model: BertModel,
            epsilon: float = 1e-5,
            alpha: float = 1e-1,
            eta: float = 1e-3,
            sigma: float = 1e-5,
            K: int = 10
    ):
        super(proximateGD, self).__init__()
        self.embed_backup = {}
        self.grad_backup = {}
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.eta = eta
        self.sigma = sigma
        self.K = K

    def save_gradients(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad():
                self.grad_backup[name] = param.grad.clone()


    def save_embeddings(self, emb_name = 'embeddings.'):
        for name, param in self.model.named_parameters():
            if param.requires_grad() and emb_name in name:
                self.embed_backup[name] = param.data.clone()


    def restore_gradients(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.grad_backup
                param.grad = self.grad_backup[name]

    def restore_embeddings(self, emb_name = 'embeddings.'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.embed_backup
                param.data = self.embed_backup
        self.embed_backup = {}

    def generate_noise(self, emb_name = 'embeddings.'):
        for name, param in self.model.named_parameters():
            if param.requires_grad() and emb_name in name:
                noise = param.data.new(param.size()).normal_(0,1) * self.sigma
                param.data.add_(noise)
                param.data = self.project(name, param.data)

    def project(self, param_name, param_data):
        change = param_data - self.embed_backup[param_name]
        change = torch.norm(change, min = - self.epsilon, max = self.epsilon)
        return self.embed_backup[param_name] + change

    def emb_ascent(self, emb_name = 'embeddings.'):
        for name, param in self.named_parameters():
            if param.requires_grad and emb_name in name:
                norm = torch.norm(param.grad, p = 'inf')
                if norm != 0 and not torch.isnan(norm):
                    param.data.add_(self.eta * (param.grad / norm))
                    param.data = self.project(name, param.data)

    def symmetric_kl(self, inputs, target):
        loss = torch.F.kl_div(torch.F.log_softmax(inputs, dim = -1), 
                              torch.F.log_softmax(target, dim = -1), 
                              reduction = 'batchmean')
        loss += torch.F.kl_div(torch.F.log_softmax(target, dim = -1),
                               torch.F.log_softmax(inputs, dim = -1),
                               reduction = 'batchmean')
        return loss*self.alpha

    def max_loss_reg(self, b_ids, b_mask, logits, emb_name = 'embeddings.'):

        #Save original gradients
        self.save_gradients()

        #Save original embeddings
        self.save_embeddings(emb_name = emb_name)

        #Update each embedding with a noise
        self.generate_noise(emb_name = emb_name)

        #For number of iterations
            #calculate new logits with noise
            #calculate loss and new gradients (with respect to just the embeddings)
            #update the noise with gradient clipping

        for k in range(self.K):
            #Zero out gradients
            self.model.zero_grad()

            #Calculate new logits with new embeddings (noise or ascent)
            adv_logits = self.model(b_ids, b_mask)
            adv_loss = self.symmetric_kl(adv_logits, logits.detach(), reduce = True)

            #Calculate new gradients
            adv_loss.backward()

            #now gradients for parameter should be updated, so we just need to normalize it, ascend, and project
            self.emb_ascent(emb_name = emb_name)

        #Restore original gradients
        self.restore_gradients()

        #Calculate the final loss as impolied by the adversarial regularizer.
        adv_logits = self.model(b_ids, b_mask)
        adv_loss = self.symmetric_kl(adv_logits, logits.detach(), reduce = True)

        #Restore to the original embeddigns
        self.restore_embeddings(emb_name = emb_name)

        return adv_loss

