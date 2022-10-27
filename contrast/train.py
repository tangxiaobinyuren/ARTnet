# -*- codding: utf-8 -*-
'''
@Author : Yuren
@Dare   : 2021/12/21-11:27 上午
'''

from __future__ import print_function, division
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
from data import Data_Loader
import torch.utils.data as Data
from model import Prl

train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available. Training on CPU')
else:
    print('CUDA is available. Training on GPU')

device = torch.device("cuda" if train_on_gpu else "cpu")


train_path = 'your_trian_data_img_path'
train_label = 'your_trian_data_label_path'
test_path = 'your_test_data_img_path'
test_label = 'your_test_data_label_path'

batch_size = 2
dtype = torch.cuda.FloatTensor

model = Prl()
model.to(device)


datalist = Data_Loader(train_path,train_label)
testlist = Data_Loader(test_path,test_label)
train_iter = Data.DataLoader(datalist,batch_size,shuffle=True)
test_iter = Data.DataLoader(testlist,batch_size,shuffle=True)

torch.cuda.empty_cache()
class CrossLoss(nn.Module):
    def __init__(self):
        super(CrossLoss, self).__init__()
        self.croloss=nn.CrossEntropyLoss()
    def forward(self,x,y):
        lossall = 0
        for i in range(4):
            yy = torch.LongTensor([y[0][i],y[1][i]])
            lossall +=self.croloss(x[i],yy)
        return lossall/4
Cross_Loss=CrossLoss()

def acc(out,y):
    theacc = 0.0
    for i in range(4):
        yy = torch.LongTensor([y[0][i], y[1][i]])
        print(out[i].argmax(dim=1))
        print(yy)
        theacc+=(out[i].argmax(dim=1) == yy).float().sum().item()
    return theacc / 8

def train(model, G_criterion, optimizer, n_epoch, train_loader, valid_loader):

    adv_loss = 10**8
    for epoch in range(n_epoch):
        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0
        model.train()
        for i, data in enumerate(train_iter, 0):
            x, y = data
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = Cross_Loss(out,y)
            loss.backward()
            optimizer.step()
            train_loss += loss.cpu().item()
            train_acc += acc(out,y)

            print('\nEpoch: {} \tStep: {} \tTraining Loss: {:.6f} \tTranining Accuracy: {:.6f}'
                  .format(epoch + 1, i + 1, train_loss / (i + 1),train_acc/(i+1)))
            del x, y, out, loss
            torch.cuda.empty_cache()


        model.eval()
        with torch.no_grad():
            for x, y in valid_loader:
                x, y = x.to(device), y.to(device)

                out= model(x)

                loss = Cross_Loss(out,y)

                valid_loss += loss.cpu().item()
                valid_acc += acc(out,y)
                del x, y, out,loss
                torch.cuda.empty_cache()
            valid_acc =  valid_acc / 1010
            valid_loss = valid_loss /1010
        print('Epoch: {}/{} \tValidation Loss: {:.6f} \tValidation Accuracy: {:.6f}'
              .format(epoch + 1, n_epoch, valid_loss,valid_acc))
        if valid_loss < adv_loss:
            adv_loss = valid_loss
            torch.save(model.state_dict(), 'your_save_model_path.pth',_use_new_zipfile_serialization=True)

    print('Finished Training')


if __name__ == '__main__':
    n_epoch = 80
    G_criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, betas=(0.90, 0.999), eps=1e-8, weight_decay=2e-5,amsgrad=True)
    train(model, G_criterion, optimizer, n_epoch, train_iter, test_iter)



