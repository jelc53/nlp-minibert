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


class MBPP(object):
    """Momentum Bregman Proximal Point Optimization (or 'Mean Teacher')
    Source: https://arxiv.org/pdf/1703.01780.pdf"""

    def __init__(self,
                 model: BertModel,
                 beta: float = 0.99,
                 mu: float = 1
    ):
        self.model = model
        self.beta = beta
        self.mu = mu
        self.theta_state = {}
        for name, param in self.model.named_parameters():
            self.theta_state[name] = param.data

    def apply_momentum(self, named_parameters):
        for name, param in named_parameters:
            self.theta_state[name] = (1-self.beta) * param.data.clone() + self.beta * self.theta_state[name]

    def bregman_divergence(self, batch, logits, task_name='default'):
        theta_prob = F.softmax(logits, dim=-1)

        param_bak = {}
        for name, param in self.model.named_parameters():
            param_bak[name] = param.data.clone()
            param.data = self.theta_state[name]

        with torch.no_grad():
            logits = model_prediction(self.model, batch, task_name)  # self.model.predict_sentiment(*batch)
            theta_til_prob = F.softmax(logits, dim=-1)

        for name, param in self.model.named_parameters():
            param.data = param_bak[name]

        l_s = 0
        if task_name != 'sts':
            l_s = F.kl_div(theta_prob.log(), theta_til_prob, reduction='batchmean') + \
            F.kl_div(theta_til_prob.log(), theta_prob, reduction='batchmean')
        else:
            l_s = torch.mean((theta_prob - theta_til_prob)**2)

        return self.mu * l_s
