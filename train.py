# coding: utf-8
# Author：WangTianRui
# Date ：2020/9/29 21:47
from hparams import hparams
from dataset import TIMIT_Dataset
import wav_loader as loader
import net_config as net_config
import pickle
from torch.utils.data import DataLoader
import module as model_cov_bn
from si_snr import *
import train_utils
import os

########################################################################
save_file = "./logs"  # model save
# 获取模型参数
para = hparams()
########################################################################

train_dataset = loader.WavDataset(para.file_scp, frame_dur=37.5)
test_dataset = loader.WavDataset(para.file_scp, frame_dur=37.5)
train_dataloader = DataLoader(train_dataset, batch_size=para.load_batch, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=para.load_batch, shuffle=True)

m_model = model_cov_bn.DCCRN_(
    n_fft=512, hop_len=int(6.25 * 16000 / 1000), net_params=net_config.get_net_params(), batch_size=para.batch_size,
    device=para.device, win_length=int((25 * 16000 / 1000))).to(para.device)
dccrn = m_model.to(para.device)
optimizer = torch.optim.Adam(dccrn.parameters(), para.learning_rate)
criterion = SiSnr()
train_utils.train(model=dccrn, optimizer=optimizer, criterion=criterion, train_iter=train_dataloader, test_iter=test_dataloader, max_epoch=5, device=para.device, batch_size=para.batch_size, log_path=save_file,
just_test=False)
