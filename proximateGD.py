import torch
import torch.nn.functional as F

from bert import BertModel


def model_prediction(model, batch, task_name='default'):
    return {
        'default': lambda: model(*batch),
        'sst': lambda: model.predict_sentiment(*batch),
        'para': lambda: model.predict_paraphrase(*batch),
        'sts': lambda: model.predict_similarity(*batch),
    }[task_name]()


class AdversarialReg(object):
    """Smoothness-inducing adversarial regularization"""
    def __init__(self,
                 model: BertModel,
                 epsilon: float = 1e-5,
                 lambda_: float = 5,
                 eta: float = 1e-3,
                 sigma: float = 1e-5,
                 K: int = 1
    ):
        super(AdversarialReg, self).__init__()
        self.embed_backup = {}
        self.grad_backup = {}
        self.model = model
        self.epsilon = epsilon
        self.lambda_ = lambda_
        self.eta = eta
        self.sigma = sigma
        self.K = K

    def save_gradients(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if param.grad == None:
                  self.grad_backup[name] = None
                else:
                  self.grad_backup[name] = param.grad.clone()

    def save_embeddings(self, emb_name):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.embed_backup[name] = param.data.clone()

    def restore_gradients(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.grad_backup
                param.grad = self.grad_backup[name]

    def restore_embeddings(self, emb_name):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.embed_backup
                param.data = self.embed_backup[name]
        self.embed_backup = {}

    def generate_noise(self, emb_name):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                noise = param.data.new(param.size()).normal_(0,1) * self.sigma
                param.data.add_(noise)
                #param.data = self.project(name, param.data)

    def project(self, param_name, param_data):
        change = param_data - self.embed_backup[param_name]
        change = torch.clamp(change, min = - self.epsilon, max = self.epsilon)
        return self.embed_backup[param_name] + change

    def emb_ascent(self, emb_name):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                norm = torch.norm(param.grad, p = float('inf'))
                if norm != 0 and not torch.isnan(norm):
                    param.data.add_(self.eta * (param.grad / norm))
                    param.data = self.project(name, param.data)

    def symmetric_kl(self, inputs, target):
        loss = F.kl_div(F.log_softmax(inputs, dim = -1), 
                        F.log_softmax(target, dim = -1), 
                        reduction = 'batchmean', 
                        log_target = True)

        loss += F.kl_div(F.log_softmax(target, dim = -1),
                         F.log_softmax(inputs, dim = -1),
                         reduction = 'batchmean', 
                         log_target = True)
        return loss*self.lambda_

    def symmetric_kl_check(self, inputs, target, reduce = True):
        epsilon = 1e-6
        bs = inputs.size(0)
        p = F.log_softmax(inputs, 1).exp()
        y = F.log_softmax(target, 1).exp()
        rp = -(1.0 / (p + epsilon) - 1 + epsilon).detach().log()
        ry = -(1.0 / (y + epsilon) - 1 + epsilon).detach().log()
        if reduce:
            return ((p * (rp - ry) * 2).sum() / bs)*self.lambda_
        else:
            return ((p * (rp - ry) * 2).sum())*self.lambda_


    def max_loss_reg(self, batch, logits, emb_name = 'embedding.', task_name='default'):

        #Overwrite current logits and do not do any dropout
        self.model.eval()
        logits = model_prediction(self.model, batch, task_name)
      
        #Save original gradients
        self.save_gradients()

        #Save original embeddings
        self.save_embeddings(emb_name)

        #Update each embedding with a noise
        self.generate_noise(emb_name)


        for _ in range(self.K):
            #Zero out gradients
            self.model.zero_grad()

            #Calculate new logits with new embeddings (noise or ascent)
            adv_logits = model_prediction(self.model, batch, task_name)
            adv_loss = self.symmetric_kl(adv_logits, logits.detach())

            #Calculate new gradients
            adv_loss.backward()

            #now gradients for parameter should be updated, so we just need to normalize it, ascend, and project
            self.emb_ascent(emb_name)

        #Restore original gradients
        self.restore_gradients()

        #Calculate the final loss as implied by the adversarial regularizer.
        adv_logits = model_prediction(self.model, batch, task_name)  # self.model.predict_sentiment(b_ids, b_mask)
        adv_loss = self.symmetric_kl(adv_logits, logits.detach())

        #Restore to the original embeddigns
        self.restore_embeddings(emb_name)

        #Allow dropout again
        self.model.train()

        return adv_loss

