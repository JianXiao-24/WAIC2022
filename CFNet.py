# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 17:13:10 2022

@author: WiCi
"""

'''
In this example, the CSI feedback based on received pilot signal is solved step by step
  - step-1: Train an AI model for channel estimation
  - step-2: Calculate eigenvectors in subband level based on estimated channel
  - step-3: Train an AI model (encoder + decoder) for CSI compression and reconstruction
'''
#=======================================================================================================================
#=======================================================================================================================
# Package Importing
import time
import gc
import numpy as np
import torch
import torch.nn as nn
# from modelDesign_example2 import *
from modelDesign import *

#=======================================================================================================================
#=======================================================================================================================

# Calculating Eigenvectors based on Estimated Channel
def cal_eigenvector(channel):
    """
        Description:
            calculate the eigenvector on each subband
        Input:
            channel: np.array, channel in frequency domain,  shape [batch_size, rx_num, tx_num, subcarrier_num]
        Output:
            eigenvectors:  np.array, eigenvector for each subband, shape [batch_size, tx_num, subband_num]
    """
    subband_num = 13
    hf_ = np.transpose(channel, [0,3,1,2]) # (batch,subcarrier,4,32)
    hf_h = np.conj(np.transpose(channel, [0,3,2,1])) # (batch,subcarrier,32,4)
    R = np.matmul(hf_h, hf_) # (batch,subcarrier,32,32)
    R = R.reshape(R.shape[0], subband_num, -1, R.shape[2],R.shape[3]).mean(axis=2) # average the R over each subband, shape:(batch,subband,32,32)
    [D,V] = np.linalg.eig(R) 
    v = V[:,:,:,0]
    eigenvectors = np.transpose(v,[0,2,1])
    return eigenvectors

def NMSE(x, x_hat):
    x_real = np.reshape(x[:, :, :, 0], (len(x), -1))
    x_imag = np.reshape(x[:, :, :, 1], (len(x), -1))
    x_hat_real = np.reshape(x_hat[:, :, :, 0], (len(x_hat), -1))
    x_hat_imag = np.reshape(x_hat[:, :, :, 1], (len(x_hat), -1))
    x_C = x_real  + 1j * (x_imag )
    x_hat_C = x_hat_real  + 1j * (x_hat_imag )
    power = np.sum(abs(x_C) ** 2, axis=1)
    mse = np.sum(abs(x_C - x_hat_C) ** 2, axis=1)
    nmse = np.mean(mse / power)
    return nmse

def cos_sim(vector_a, vector_b):
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = vector_a * vector_b.H
    num1 = np.sqrt(vector_a * vector_a.H)
    num2 = np.sqrt(vector_b * vector_b.H)
    cos = (num / (num1*num2))
    return cos.item()
def cal_score(w_true,w_pre,NUM_SAMPLES,NUM_SUBBAND):
    w_true = np.transpose(w_true, [0, 3, 2, 1])
    w_pre = np.transpose(w_pre, [0, 3, 2, 1])
    img_total = NUM_TX * 2
    num_sample_subband = NUM_SAMPLES * NUM_SUBBAND
    W_true = np.reshape(w_true, [num_sample_subband, img_total])
    W_pre = np.reshape(w_pre, [num_sample_subband, img_total])
    W_true2 = W_true[0:num_sample_subband, 0:int(img_total):2] + 1j*W_true[0:num_sample_subband, 1:int(img_total):2]
    W_pre2 = W_pre[0:num_sample_subband, 0:int(img_total):2] + 1j*W_pre[0:num_sample_subband, 1:int(img_total):2]
    score_cos = 0
    for i in range(num_sample_subband):
        W_true2_sample = W_true2[i:i+1,]
        W_pre2_sample = W_pre2[i:i+1,]
        score_tmp = cos_sim(W_true2_sample,W_pre2_sample)
        score_cos = score_cos + abs(score_tmp)*abs(score_tmp)
    score_cos = score_cos/num_sample_subband
    return score_cos

import math
def adjust_learning_rate(optimizer, epoch,learning_rate_init,learning_rate_final):
    """For resnet, the lr starts from 0.1, and is divided by 10 at 80 and 120 epochs"""
    # learning_rate_init = 1e-4
    # learning_rate_final = 1e-7
    epochs =EPOCHS
    lr = learning_rate_final + 0.5*(learning_rate_init-learning_rate_final)*(1+math.cos((epoch*3.14)/epochs))
    # lr = 0.00003* (1+math.cos(float(epoch)/TOTAL_EPOCHS*math.pi))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

# Parameters Setting
NUM_FEEDBACK_BITS = 64
NUM_RX = 4
NUM_TX = 32
NUM_SUBBAND = 13
BATCH_SIZE = 256
EPOCHS = 250
LEARNING_RATE = 6e-5
model_ce_path = './modelSubmit/encModel_p1_1.pth.tar'
model_encoder_path = './modelSubmit/encModel_p1_2.pth.tar'
model_decoder_path = './modelSubmit/decModel_p1_1.pth.tar'
#=======================================================================================================================
#=======================================================================================================================
# Data Loading
# load pilot
pilot = np.load('D:/RL4Com/RL4Com/Pytorch_Template/pilot_1.npz')['data']
pilot = np.expand_dims(pilot, axis=1)
pilot = np.concatenate([np.real(pilot), np.imag(pilot)], 1)
# # load eigenvectors
w = np.load('D:/RL4Com/RL4Com/Pytorch_Template/w.npz')['data']
w = np.expand_dims(w, axis=1)
w = np.concatenate([np.real(w), np.imag(w)], 1)
# # load channel in time domain
# ht = np.load('../dataset/h_time.npz')['data']
hf = np.load('D:/RL4Com/RL4Com/Pytorch_Template/hf.npz')['data']
hf = np.expand_dims(hf, axis=1)
hf = np.concatenate([np.real(hf), np.imag(hf)], 1)
print(pilot.shape, w.shape, hf.shape)
print('data loading is finished ...')

'''
# select part of samples for test
sample_num = 10000
pilot, w, ht = pilot[:sample_num,...], w[:sample_num,...], ht[:sample_num,...]
print(pilot.shape, w.shape, ht.shape)
'''

#=======================================================================================================================
#=======================================================================================================================
# Generate Label for Channel Estimation 
# following is an example of generating label of channel estimation (transform channel from time to frequency domain)
# subcarrierNum = 52*12 # 52 resource block, 12 subcarrier per resource block
# estimatedSubcarrier = np.arange(0, subcarrierNum, 12) # configure the channel on which subcarriers to be estimated (this example only estimate channel on part subcarriers)
# FFTsize = 1024
# delayNum = ht.shape[-1]
# batch_size = 100
# ht_batch = ht.reshape(-1, batch_size, ht.shape[1], ht.shape[2], ht.shape[3])
# hf_batch = np.zeros([ht_batch.shape[0], batch_size, ht.shape[1], ht.shape[2], len(estimatedSubcarrier)], dtype='complex64')
# for i in range(ht_batch.shape[0]):
#     print("CE Label Generation Progress: {}/{}".format(i+1, ht_batch.shape[0]), end="\r", flush = True)
#     ht_ = ht_batch[i,...]
#     # padding
#     ht_ = np.pad(ht_, ((0,0),(0,0),(0,0),(0,FFTsize-delayNum)))
#     # FFT
#     hf_ = np.fft.fftshift(np.fft.fft(ht_), axes=(3,))
#     startSample, endSample = int(FFTsize/2-subcarrierNum/2), int(FFTsize/2+subcarrierNum/2)
#     hf_all_subc = hf_[...,startSample:endSample] # channel on all subcarrier
#     hf_batch[i,...] = hf_all_subc[...,estimatedSubcarrier] # save channel on selected subcarrier 
# print(hf_batch.shape, hf_batch.dtype)
# hf = hf_batch.reshape(-1, hf_batch.shape[2], hf_batch.shape[3], hf_batch.shape[4])
# del ht, ht_batch, hf_batch
# gc.collect()
# hf = np.expand_dims(hf, axis=1)
# hf = np.concatenate([np.real(hf), np.imag(hf)], 1)
# print('\n', hf.shape)
# print('label generation for channel estimation is finished ...')
#=======================================================================================================================
#=======================================================================================================================
# Channel Estimation Model Training and Saving
if __name__ == '__main__':
    subc_num = pilot.shape[3]
    model_ce = channel_est(subc_num).cuda()
    model_ce.load_state_dict(torch.load(model_ce_path)['state_dict'])
    
    criterion = nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(model_ce.parameters(), lr=LEARNING_RATE)
    # data loading
    split_idx = int(0.9 * pilot.shape[0])
    pilot_train, pilot_test = pilot[:split_idx,...], pilot[split_idx:,...]
    hf_train, hf_test = hf[:split_idx,...], hf[split_idx:,...]
    train_dataset = DatasetFolder(pilot_train, hf_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    test_dataset = DatasetFolder(pilot_test, hf_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    # model training and saving
    model_ce.eval()
    testLoss = 0
    testNMSE = 0
    
    with torch.no_grad():
        for i, (modelInput, label) in enumerate(test_loader):
            modelInput, label = modelInput.cuda(), label.cuda()
            modelOutput = model_ce(modelInput)
            testLoss += criterion(label, modelOutput).item() * modelInput.size(0)
            testNMSE1 = NMSE(modelOutput.cpu().data.numpy(),label.cpu().data.numpy())
            testNMSE += 10*np.log(testNMSE1)/np.log(10)
            if i% 50 ==0:
                trainNMSE = NMSE(modelOutput.cpu().data.numpy(),label.cpu().data.numpy())
                print('Iter : %d/%d, TestNMSE: %.6f' % (
                  i + 1, len(test_dataset) // BATCH_SIZE,10*np.log(trainNMSE)/np.log(10)))
            
        avgTestLoss = testLoss / len(test_dataset)
        avgTrainLoss = testNMSE / len(test_loader)
        print('Train Loss:{loss1:.5f}\t' 'Val Loss:{loss2:.5f}\t'.format(loss1=avgTrainLoss, loss2=avgTestLoss))
        # if avgTestLoss < bestLoss:
        #     torch.save({'state_dict': model_ce.state_dict(), }, model_ce_path)
        #     bestLoss = avgTestLoss
        #     print("Model saved")
    del hf 
    del hf_train 
    del hf_test 
    del train_loader 
    del test_loader 
    del test_dataset 
    del train_dataset
    gc.collect()
    print('model training for channel estimation is finished ...')
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    #=======================================================================================================================
    #=======================================================================================================================
    
    w_pre = []
    model_ce.load_state_dict(torch.load(model_ce_path)['state_dict'])
    model_ce.eval()
    test_dataset = DatasetFolder_eval(pilot)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            print("Eigenvectors Calculation Progress: {}/{}".format(idx+1, len(test_loader)), end="\r", flush = True)
            data = data.cuda()
            # step 1: channel estimation
            h = model_ce(data) # (batch,2,4,32,52)
            # step 2: eigenvector calculation
            h_complex = h[:,0,...] + 1j*h[:,1,...] # (batch,4,32,52)
            h_complex = h_complex.cpu().numpy()
            v = cal_eigenvector(h_complex)
            w_complex = torch.from_numpy(v)
            w_tmp = torch.zeros([h.shape[0], 2, 32, 13], dtype=torch.float32).cuda() # (batch,2,32,13)
            w_tmp[:,0,:,:] = torch.real(w_complex)
            w_tmp[:,1,:,:] = torch.imag(w_complex)
            w_tmp = w_tmp.cpu().numpy()
            if idx == 0:
                w_pre = w_tmp
            else:
                w_pre = np.concatenate((w_pre, w_tmp), axis=0)
    del pilot
    gc.collect()
    print('\n', w_pre.shape)
    print('eigenvectors calculation based on estimated channel is finished ...')
    np.save('wpre2.npy',w_pre)
    w_pre = np.load('wpre2.npy')
    #=======================================================================================================================
    #=======================================================================================================================
    # CSI feedback Model Training and Saving
    model_fb = AutoEncoder(NUM_FEEDBACK_BITS).cuda()
    # criterion = nn.MSELoss().cuda()
    criterion = CosineSimilarityLoss().cuda()
    optimizer = torch.optim.AdamW(model_fb.parameters(), lr=LEARNING_RATE)
    # data loading
    split_idx = int(0.9 * w_pre.shape[0])
    wpre_train, wpre_test = w_pre[:split_idx,...], w_pre[split_idx:,...]
    w_train, w_test = w[:split_idx,...], w[split_idx:,...]
    train_dataset = DatasetFolder_mixup(wpre_train, w_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    test_dataset = DatasetFolder(wpre_test, w_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    bestLoss = 10
    for epoch in range(EPOCHS):
        lr = adjust_learning_rate(optimizer, epoch,6e-5,2e-5)
        # print(lr)
        start = time.time()
        model_fb.train()
        trainLoss = 0
        for i, (modelInput, label) in enumerate(train_loader):
            modelInput, label = modelInput.cuda(), label.cuda()
            
            mix_input, mix_label = cutmixup(
                    modelInput, label,    
                    mixup_prob=0.2, mixup_alpha=1.2,
                    cutmix_prob=0.4, cutmix_alpha=0.7
                )
            
            mix_input, mix_label = mixup(mix_input, mix_label, prob=0.4, alpha=0.4)
            
            # mix_input, mix_label = cutmix(mix_input, mix_label, prob=0.4, alpha=0.7)
            
            # modelInput, label = rgb(mix_input, mix_label, prob=0.4)
            modelInput, label = rgb1(mix_input, mix_label, prob=0.4)
            
            modelOutput = model_fb(modelInput)
            loss = criterion(label, modelOutput)
            trainLoss += loss.item() * modelInput.size(0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i% ((int(split_idx//BATCH_SIZE))//5) ==0:
                a,b,c,d = modelOutput.size()
                
                score1 = cal_score(modelOutput.cpu().detach().numpy(), label.cpu().detach().numpy(),a,d)
                print('The training score 1 is ' + str(score1))
            
        avgTrainLoss = trainLoss / len(train_dataset)
        model_fb.eval()
        testLoss = 0
        with torch.no_grad():
            for i, (modelInput, label) in enumerate(test_loader):
                modelInput, label = modelInput.cuda(), label.cuda()
                modelOutput = model_fb(modelInput)
                testLoss += criterion(label, modelOutput).item() * modelInput.size(0)
                if i% 50 ==0:
                    a,b,c,d = modelOutput.size()
                
                    score1 = cal_score(modelOutput.cpu().detach().numpy(), label.cpu().detach().numpy(),a,d)
                    print('The test score 1 is ' + str(score1))
                    
            avgTestLoss = testLoss / len(test_dataset)
            # print('Epoch:[{0}]\t' 'Train Loss:{loss1:.5f}\t' 'Val Loss:{loss2:.5f}\t' 'Time:{time:.1f}secs\t'.format(epoch, loss1=avgTrainLoss, loss2=avgTestLoss, time=time.time()-start))
            if avgTestLoss < bestLoss:
                # Model saving
                # Encoder Saving
                torch.save({'state_dict': model_fb.encoder.state_dict(), }, model_encoder_path)
                # Decoder Saving
                torch.save({'state_dict': model_fb.decoder.state_dict(), }, model_decoder_path)
                print("Model saved")
                bestLoss = avgTestLoss
            print('Epoch:[{0}]\t' 'Train Loss:{loss1:.5f}\t' 'Val Loss:{loss2:.5f}\t' 'Best Loss:{loss3:.5f}\t' 'Time:{time:.1f}secs\t'.format(epoch, loss1=avgTrainLoss, loss2=avgTestLoss, loss3=bestLoss, time=time.time()-start))
    print('model training for CSI feedback is finished ...')
    print('Training is finished!')