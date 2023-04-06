import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from mir.common import WORKING_PATH
import os
import numpy as np

class NetworkBehavior(nn.Module):

    def __init__(self):
        super().__init__()
        self.use_gpu=torch.cuda.device_count()>0
        self.use_data_parallel=False

    def get_optimizer(self):
        return optim.Adam(self.parameters())

    def forward(self, *args):
        raise NotImplementedError()

    def init_settings(self, is_training):
        if(self.use_gpu):
            self.cuda()
        else:
            self.cpu()
        if(is_training):
            self.train()
        else:
            self.eval()
        if(self.use_data_parallel):
            self.parallel_net=[nn.DataParallel(self)]

    def feed(self, *args):
        if(self.use_data_parallel):
            return self.parallel_net[0](*args)
        else:
            return self(*args)

    def loss(self, *args):
        raise NotImplementedError()

    def inference(self, *args):
        raise NotImplementedError()

    def evaluation(self, *args):
        raise NotImplementedError()

class NetworkInterface:

    def __init__(self, net, save_name, load_checkpoint=False, load_path='cache_data'):
        self.net=net
        if(not isinstance(self.net,NetworkBehavior)):
            raise Exception('Invalid network type')
        if('(p)' in save_name):
            self.net.use_data_parallel=True
        self.net.init_settings(False)
        self.save_name=save_name
        save_path=os.path.join(WORKING_PATH,load_path,'%s.sdict'%save_name)
        cp_save_path=os.path.join(WORKING_PATH,load_path,'%s.cp.sdict'%save_name)
        self.finalized=False
        self.optimizer=self.net.get_optimizer()
        self.counter=0
        self.best_val_loss=np.inf
        self.best_epoch_dist=0
        if(os.path.exists(save_path)):
            state_dict=torch.load(save_path,map_location='cuda' if self.net.use_gpu else 'cpu')
            # The following codes are for torch 4.0 compatibility
            # new_state_dict={}
            # for key in state_dict['net']:
            #     if('num_batches_tracked' not in key):
            #         new_state_dict[key]=state_dict['net'][key]
            # self.net.load_state_dict(new_state_dict)
            self.net.load_state_dict(state_dict['net'])
            self.counter=state_dict['counter']
            self.optimizer.load_state_dict(state_dict['opt'])
            try:
                self.best_epoch_dist=state_dict['best_epoch_dist']
                self.best_val_loss=state_dict['best_val_loss']
            except:
                pass
            self.finalized=True
        elif(load_checkpoint and os.path.exists(cp_save_path)):
            state_dict=torch.load(cp_save_path,map_location='cuda' if self.net.use_gpu else 'cpu')
            # The following codes are for torch 4.0 compatibility
            # new_state_dict={}
            # for key in state_dict['net']:
            #     if('num_batches_tracked' not in key):
            #         new_state_dict[key]=state_dict['net'][key]
            # self.net.load_state_dict(new_state_dict)
            self.net.load_state_dict(state_dict['net'])
            self.counter=state_dict['counter']
            self.optimizer.load_state_dict(state_dict['opt'])
            try:
                self.best_epoch_dist=state_dict['best_epoch_dist']
                self.best_val_loss=state_dict['best_val_loss']
            except:
                pass

    def train_supervised(self, train_set, val_set, batch_size, val_batch_size=None, eval_set=None,
                         learning_rates_dict=1e-3, round_per_print=20, learning_rate_decay=0.0,
                         round_per_val=100, round_per_save=500, early_end_epochs=10):
        if(self.finalized):
            print('Model is already finalized. To perform new training, delete the save file.')
            return
        print('Training model %s (%s)'%(self.net.__class__.__name__,self.save_name))
        if(isinstance(learning_rates_dict,dict)):
            learning_rates_list=sorted(learning_rates_dict.items(),reverse=True)
        else:
            learning_rates_list=[(learning_rates_dict,1.0)]
        # TODO: multi-thread support
        if(train_set.num_workers>0):
            # todo: unload to prevent unpickable situations
            pass
        train_set_loader=DataLoader(train_set,batch_size=batch_size,shuffle=train_set.need_shuffle,
                                    num_workers=train_set.num_workers,worker_init_fn=train_set.__init_training_worker,
                                    collate_fn=val_set.collate_fn)
        if(train_set.num_workers==0):
            train_set.init_worker(-1,True)
        if(val_set.num_workers>0):
            # todo: unload to prevent unpickable situations
            pass
        val_set_loader=DataLoader(val_set,batch_size=batch_size if val_batch_size is None else val_batch_size,shuffle=val_set.need_shuffle,
                                  num_workers=val_set.num_workers,worker_init_fn=val_set.__init_validation_worker,
                                  collate_fn=val_set.collate_fn)
        if(val_set.num_workers==0):
            val_set.init_worker(-1,False)
        if(round_per_val>=0):
            val_set_iter=iter(val_set_loader)
        learning_rates=[x[0] for x in learning_rates_list]
        learning_rates_batch_count=[int(np.ceil(len(train_set)*x[1]/batch_size)) if x[1]>0
                                    else -int(x[1]) for x in learning_rates_list]
        print(len(train_set),'samples','batch_size =',batch_size)
        for (learning_rate,batch_count) in zip(learning_rates,learning_rates_batch_count):
            print(batch_count,'mini batches for learning rate =',learning_rate,flush=True)
        current_counter=0
        self.net.init_settings(True)
        for (learning_rate,batch_count) in zip(learning_rates,learning_rates_batch_count):
            i=0
            if(current_counter<self.counter):
                del_count=min(self.counter-current_counter,batch_count)
                i+=del_count
                current_counter+=del_count
            if(i<batch_count):
                for param_group in self.optimizer.param_groups:
                    param_group['lr']=learning_rate
                    param_group['lr_decay']=learning_rate_decay
                running_loss=np.array([0.0])
                running_loss_count=0
                self.best_epoch_dist=0
                while(i<batch_count):
                    train_set_iter=iter(train_set_loader)
                    while True:
                        # print('Data fetch begin')
                        try:
                            input_tuple=next(train_set_iter)
                        except StopIteration:
                            break

                        # print('Data fetch end')
                        # todo: remove this and put it in collate functions
                        def send_to_gpu(var):
                            if(isinstance(var,list)):
                                return [send_to_gpu(sub_var) for sub_var in var]
                            if(isinstance(var,tuple)):
                                return tuple(send_to_gpu(sub_var) for sub_var in var)
                            return var.cuda()
                        if(self.net.use_gpu):
                            input_tuple=send_to_gpu(input_tuple)
                        self.optimizer.zero_grad()
                        raw_loss=self.net.loss(*input_tuple)
                        if(isinstance(raw_loss,tuple)):
                            loss=torch.sum(torch.stack(raw_loss))
                            running_loss=running_loss+np.array([x.item() for x in raw_loss])
                        else:
                            loss=raw_loss
                            running_loss+=loss.item()
                        running_loss_count+=1
                        if(loss.grad_fn is None):
                            print('Warning: the loss of the batch does not have a grad_fn. Ignored.')
                        else:
                            loss.backward()
                            self.optimizer.step()
                        if(i%round_per_print==round_per_print-1):
                            if(len(running_loss)>1):
                                loss_str='%.6f(%s)'%(running_loss.sum()/running_loss_count,
                                    ','.join('%.6f'%(x/running_loss_count) for x in running_loss))
                            else:
                                loss_str='%.6f'%(running_loss.sum()/running_loss_count)
                            print('[%f, %.2f%% (%d/%d)] loss: %s' %
                                  (learning_rate,(i+1)/batch_count*100,i+1,batch_count,loss_str),flush=True)
                            running_loss=np.array([0.0])
                            running_loss_count=0
                        if(i%round_per_val==round_per_val-1):
                            val_loss=np.array([0.0])
                            for j in range(round_per_print):
                                try:
                                    val_input_tuple=next(val_set_iter)
                                except StopIteration:
                                    val_set_iter=iter(val_set_loader)
                                    val_input_tuple=next(val_set_iter)
                                if(self.net.use_gpu):
                                    val_input_tuple=send_to_gpu(val_input_tuple)
                                self.net.eval()
                                with torch.no_grad():
                                    raw_val_loss=self.net.loss(*val_input_tuple)
                                    if(isinstance(raw_val_loss,tuple)):
                                        val_loss=val_loss+np.array([x.item() for x in raw_loss])
                                    else:
                                        val_loss=val_loss+raw_val_loss.item()
                                self.net.train()
                            if(len(val_loss)>1):
                                loss_str='%.6f(%s)'%(val_loss.sum()/round_per_print,
                                    ','.join('%.6f'%(x/round_per_print) for x in val_loss))
                            else:
                                loss_str='%.6f'%(val_loss.sum()/round_per_print)
                            print('[%f, %.2f%% (%d/%d)] val_loss: %s' %
                                  (learning_rate,(i+1)/batch_count*100,i+1,batch_count,loss_str),flush=True)


                        if(round_per_val!=-1 and (i%round_per_save==round_per_save-1 or i+1==batch_count)):
                            if(self.net.use_gpu):
                                self.net.cpu()
                            torch.save({'net':self.net.state_dict(),
                                        'opt':self.optimizer.state_dict(),
                                        'counter':current_counter+1
                                        },os.path.join(WORKING_PATH,'cache_data/%s.cp.sdict'%self.save_name))
                            if(self.net.use_gpu):
                                self.net.cuda()
                            print('[%f, %.2f%% (%d/%d)] checkpoint created' %
                                  (learning_rate,(i+1)/batch_count*100,i+1,batch_count),flush=True)
                        i+=1
                        current_counter+=1
                        if(i==batch_count):
                            break
                    if(round_per_val==-1): # evaluation per epoch
                        val_set_iter=iter(val_set_loader)
                        j=0
                        val_loss=0.0
                        while True:
                            try:
                                val_input_tuple=next(val_set_iter)
                            except StopIteration:
                                break
                            if(self.net.use_gpu):
                                val_input_tuple=(var.cuda() for var in val_input_tuple)
                            self.net.eval()
                            with torch.no_grad():
                                raw_val_loss=self.net.loss(*val_input_tuple)
                                if(isinstance(raw_val_loss,tuple)):
                                    val_loss=val_loss+np.sum([x.item() for x in raw_loss])
                                else:
                                    val_loss=val_loss+raw_val_loss.item()
                            self.net.train()
                            if(j%(round_per_print*batch_count//val_batch_size)==0):
                                print('Validation: %d/%d'%(j,len(val_set_loader)),flush=True)
                            j+=1
                        mean_val_loss=val_loss/j
                        print('[%f, %.2f%% (%d/%d)] val_loss: %.6f best_val_loss: %.6f (%d epochs away)' %
                              (learning_rate,(i+1)/batch_count*100,i+1,batch_count,
                               mean_val_loss,self.best_val_loss,self.best_epoch_dist),flush=True)
                        if(self.net.use_gpu):
                            self.net.cpu()
                        if(mean_val_loss<self.best_val_loss):
                            self.best_val_loss=mean_val_loss
                            self.best_epoch_dist=0
                            torch.save({'loss':mean_val_loss,
                                        'learning_rate':learning_rate,
                                        'step':i,
                                        'batch_count':batch_count,
                                        'net':self.net.state_dict(),
                                        'opt':self.optimizer.state_dict(),
                                        'counter':current_counter+1
                                        },os.path.join(WORKING_PATH,'cache_data/%s.best.sdict'%self.save_name))
                        else:
                            self.best_epoch_dist+=1
                        torch.save({'best_val_loss':self.best_val_loss,
                                    'best_epoch_dist':self.best_epoch_dist,
                                    'net':self.net.state_dict(),
                                    'opt':self.optimizer.state_dict(),
                                    'counter':current_counter+1
                                    },os.path.join(WORKING_PATH,'cache_data/%s.cp.sdict'%self.save_name))
                        if(self.net.use_gpu):
                            self.net.cuda()
                        print('[%f, %.2f%% (%d/%d)] checkpoint created' %
                              (learning_rate,(i+1)/batch_count*100,i+1,batch_count),flush=True)
                        if(self.best_epoch_dist>=early_end_epochs):
                            print('Early stopping for lr %f'%learning_rate,flush=True)
                            current_counter+=batch_count-i
                            i=batch_count
                            break

        if(self.net.use_gpu):
            self.net.cpu()
        torch.save({'net':self.net.state_dict(),
                    'opt':self.optimizer.state_dict(),
                    'counter':current_counter
                    },os.path.join(WORKING_PATH,'cache_data/%s.sdict'%self.save_name))
        if(self.net.use_gpu):
            self.net.cuda()


    def inference(self, *args,**kwargs):
        self.net.init_settings(False)
        inputs=[torch.tensor(arg,dtype=torch.float if arg.dtype in [np.float16,np.float32,np.float64] else torch.long)
                for arg in args]
        if(self.net.use_gpu):
            inputs=[input.cuda() for input in inputs]
        with torch.no_grad():
            return self.net.inference(*inputs,**kwargs)

    def inference_function(self,function,*args,**kwargs):
        self.net.init_settings(False)
        inputs=[torch.tensor(arg,dtype=torch.float if arg.dtype in [np.float16,np.float32,np.float64] else torch.long)
                for arg in args]
        if(self.net.use_gpu):
            inputs=[input.cuda() for input in inputs]
        with torch.no_grad():
            return self.net.__class__.__dict__[function](self.net,*inputs,**kwargs)


