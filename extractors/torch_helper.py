import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

def log(*args,**kwargs):
    return print(*args,flush=True,**kwargs)

class Trainer():

    def __init__(self,network,model_filename,train_dataset,val_dataset,continue_training,
                 learning_rates,batches_per_learning_rate,
                 batch_size=512,loss_fun=None,optimizer=None,
                 round_per_print=20,round_per_val=200,round_per_save=500):
        self.network=network
        self.model_filename=model_filename
        self.train_dataset=train_dataset
        self.val_dataset=val_dataset
        self.continue_training=continue_training
        self.learning_rates=learning_rates
        self.batches_per_learning_rate=batches_per_learning_rate
        self.batch_size=batch_size
        if(loss_fun is None):
            self.criterion=nn.CrossEntropyLoss()
        else:
            self.criterion=loss_fun
        if(optimizer is None):
            self.optimizer=optim.Adam(self.network.parameters(),lr=self.learning_rates[0])
        else:
            self.optimizer=optimizer
        self.round_per_print=round_per_print
        self.round_per_val=round_per_val
        self.round_per_save=round_per_save

    def calc_loss(self,inputs,labels):
        return self.criterion(self.network(inputs),labels)

    def train(self):
        self.network.train()
        self.network.cuda()
        log('Creating model')
        train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True)
        val_iter=iter(val_dataloader)
        train_iter=iter(train_dataloader)
        log(len(self.train_dataset),'samples','batch_size =',self.batch_size)
        log(int(np.ceil(len(self.train_dataset)/self.batch_size)),'mini_batches per epoch')
        cp_counter=0
        last_cp=0
        if(self.continue_training):
            try:
                state_dict=torch.load('cache_data/'+self.model_filename+'.checkpoint')
                last_cp=state_dict['counter']
                self.network.load_state_dict(state_dict['net'])
                self.optimizer.load_state_dict(state_dict['opt'])
                log('Loaded checkpoint @ %d'%last_cp)
            except:
                log('Begin training new model')




def net_train(network,model_filename,
              train_dataset,val_dataset,shuffle_train_set,shuffle_val_set,
              continue_training,learning_rates,epochs_per_learning_rate=1,
              batch_size=512,train_sample_use_ratio=1.0,special_loss=''):
    network.train()
    network.cuda()
    log('Creating model')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train_set)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle_val_set)
    val_iter = iter(val_dataloader)
    optimizer = optim.Adam(network.parameters(),lr=learning_rates[0])
    log(len(train_dataset),'samples','batch_size =',batch_size)
    batch_count_limit=int(np.ceil(len(train_dataset)*train_sample_use_ratio/batch_size))
    log(batch_count_limit,'mini_batches pre epoch')
    criterion = nn.CrossEntropyLoss()

    ROUND_PER_PRINT=20
    ROUND_PER_VAL=200
    ROUND_PER_SAVE=1000
    cp_counter=0
    last_cp=0
    if(continue_training):
        try:
            state_dict=torch.load('cache_data/'+model_filename+'.checkpoint')
            last_cp=state_dict['counter']
            network.load_state_dict(state_dict['net'])
            optimizer.load_state_dict(state_dict['opt'])
            log('Loaded checkpoint @ %d'%last_cp)
        except:
            log('Begin training new model')

    for learning_rate in learning_rates:
        for param_group in optimizer.param_groups:
            param_group['lr']=learning_rate
        for epoch in range(epochs_per_learning_rate):
            running_loss = 0.0
            for i, data in enumerate(train_dataloader, 0):
                if(i>batch_count_limit):
                    break
                cp_counter+=1
                if(cp_counter<=last_cp):
                    continue
                # get the inputs
                inputs, labels = data
                inputs=inputs.cuda()
                labels=labels.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                if(special_loss==''):
                    loss=criterion(network(inputs),labels)
                elif(special_loss=='soft_binary'):
                    outputs = F.softmax(network(inputs),dim=1)
                    loss=-(torch.log(outputs[:,1]+1e-10)*labels+
                           torch.log(outputs[:,0]+1e-10)*(1-labels)).mean()
                else:
                    raise NotImplementedError()
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if(i%ROUND_PER_PRINT==ROUND_PER_PRINT-1):
                    log('[%f, %d, %5d] loss: %.6f' %
                          (learning_rate, epoch + 1, i + 1, running_loss / ROUND_PER_PRINT))
                    running_loss = 0.0

                if(i%ROUND_PER_VAL==ROUND_PER_VAL-1):
                    data=next(val_iter)
                    val_inputs, val_labels = data
                    val_inputs=val_inputs.cuda()
                    val_labels=val_labels.cuda()
                    with torch.no_grad():
                        # loss = F.mse_loss(labels,outputs)
                        if(special_loss==''):
                            val_loss=criterion(network(val_inputs),val_labels)
                        elif(special_loss=='soft_binary'):
                            val_outputs = F.softmax(network(val_inputs),dim=1)
                            val_loss=-(torch.log(val_outputs[:,1]+1e-10)*val_labels+
                                   torch.log(val_outputs[:,0]+1e-10)*(1-val_labels)).mean()
                        else:
                            raise NotImplementedError()
                        log('[%f, %d, %5d] val_loss: %.6f' %
                              (learning_rate, epoch + 1, i + 1, val_loss.item()))

                if(i%ROUND_PER_SAVE==ROUND_PER_SAVE-1):
                    network.cpu()
                    torch.save(network.state_dict(),'cache_data/'+model_filename+'.last')
                    torch.save({'net':network.state_dict(),
                                'opt':optimizer.state_dict(),
                                'counter':cp_counter
                                },'cache_data/'+model_filename+'.checkpoint')
                    network.cuda()
                    log('Checkpoint created')
    network.cpu()
    torch.save(network.state_dict(),'cache_data/'+model_filename)
    log('Model saved')
    return network
