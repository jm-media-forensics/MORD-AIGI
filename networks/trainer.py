import functools
import torch
import torch.nn as nn
from networks.resnet import resnet50
#from networks.net1 import resnet50
#from networks.resnettwopca import resnet50
from networks.base_model import BaseModel, init_weights


class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)

        if self.isTrain and not opt.continue_train:
            self.model = resnet50(pretrained=False, num_classes=1)

        if not self.isTrain or opt.continue_train:
            self.model = resnet50(num_classes=1)

        if self.isTrain:
            self.loss_fn = nn.BCEWithLogitsLoss()
            # initialize optimizers
            if opt.optim == 'adam':
                self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
            elif opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                                                 lr=opt.lr, momentum=0.0, weight_decay=0)
            else:
                raise ValueError("optim should be [adam, sgd]")

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.epoch)
        self.model.to(opt.gpu_ids[0])
 

    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= 0.9
            if param_group['lr'] < min_lr:
                return False
        self.lr = param_group['lr']
        print('*'*25)
        print(f'Changing lr from {param_group["lr"]/0.9} to {param_group["lr"]}')
        print('*'*25)
        return True

    def set_input(self, input):
        self.input = input[0].to(self.device)
        self.label = input[1].to(self.device).float()


    def forward(self):
        self.output = self.model(self.input)

    def get_loss(self):
        #return self.loss_fn(self.output.squeeze(1), self.label)
        loss = 0
        for i in range(3):
          loss += self.loss_fn(self.output[:, i], self.label)
        return loss
     #the loss of resnet
    '''
    def optimize_parameters(self):
        branch1_loss = 0
        branch2_loss = 0
        comb_loss = 0
        
            
        self.forward()
        branch1_loss = self.loss_fn(self.output[:, 0], self.label)
        branch2_loss = self.loss_fn(self.output[:, 1], self.label)
        
        comb_loss = self.loss_fn(self.output[:, 2], self.label)
        g = self.output[:, 3]
        eps = 1e-6
        ent = -(g*torch.log(g+eps) + (1-g)*torch.log(1-g+eps)).mean()
        #self.final_loss = comb_loss + 0.1 * branch1_loss + 0.1 * branch2_loss + 1e-4*ent
        #self.final_loss = comb_loss
        self.final_loss = branch1_loss +  branch2_loss
        self.optimizer.zero_grad()
        self.final_loss.backward()
        self.optimizer.step()
         
        self.loss = self.final_loss
    '''
    
    
    #new patch loss
    '''
    def optimize_parameters(self):
        branch1_loss = 0
        branch2_loss = 0
        comb_loss = 0
        
            
        self.forward()
        branch1_loss = self.loss_fn(self.output[:, 0], self.label)
        branch2_loss = self.loss_fn(self.output[:, 1], self.label)
        
        comb_loss = self.loss_fn(self.output[:, 2], self.label)

        #self.final_loss = comb_loss + 0.1 * branch1_loss + 0.1 * branch2_loss + 1e-4*ent
        #self.final_loss = comb_loss
        self.final_loss = comb_loss + 0.1 * branch1_loss +  0.1 * branch2_loss 
        self.optimizer.zero_grad()
        self.final_loss.backward()
        self.optimizer.step()
         
        self.loss = self.final_loss
    '''
  
    def optimize_parameters(self):
        self.forward()
        #self.loss = self.loss_fn(self.output.squeeze(1), self.label)
        self.loss = 0
        self.loss += self.loss_fn(self.output[:, 1], self.label)
        self.loss += self.loss_fn(self.output[:, 0], self.label)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    '''    
    def optimize_parameters(self):
        branch1_loss = 0
        branch2_loss = 0
        comb_loss = 0
        
        for param in self.model.fc_comb.parameters():
            param.requires_grad = False
            
        self.forward()
        branch1_loss = self.loss_fn(self.output[:, 0], self.label)
        self.optimizer.zero_grad()
        branch1_loss.backward()
        self.optimizer.step()
    
        
        self.forward()  
        branch2_loss = self.loss_fn(self.output[:, 1], self.label)
        self.optimizer.zero_grad()
        branch2_loss.backward()
        self.optimizer.step()
    
        
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.fc_comb.parameters():
            param.requires_grad = True
        
        self.forward()  
        comb_loss = self.loss_fn(self.output[:, 2], self.label)
        self.optimizer.zero_grad()
        comb_loss.backward()
        self.optimizer.step()
        
     
        for param in self.model.parameters():
            param.requires_grad = True
        
         
        self.loss = branch1_loss + branch2_loss + comb_loss
    '''