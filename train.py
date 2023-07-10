from hparams import hparams, get_net_params
import wav_loader as loader
from torch.utils.data import DataLoader
import module as model_cov_bn
from si_snr import *
from train_utils import my_collect, train


########################################################################
save_file = "./logs"  # model save
# 获取模型参数
para = hparams()
########################################################################

train_dataset = loader.WavDataset(para.file_scp, frame_dur=37.5)
test_dataset = loader.WavDataset(para.file_scp_test, frame_dur=37.5)
train_dataloader = DataLoader(train_dataset, batch_size=para.load_batch, shuffle=True, collate_fn=my_collect)
test_dataloader = DataLoader(test_dataset, batch_size=para.load_batch, shuffle=True, collate_fn=my_collect)

if para.checkpoint:
    m_model = torch.load(para.lastModelPath).to(para.device)
else:
    m_model = model_cov_bn.DCCRN_(n_fft=para.para_stft["N_fft"], hop_len=para.para_stft["hop_length"],
                              net_params=get_net_params(), batch_size=para.batch_size, device=para.device,
                              win_length=para.para_stft["win_length"]).to(para.device)
dccrn = m_model.to(para.device)
optimizer = torch.optim.Adam(dccrn.parameters(), para.learning_rate)
criterion = SiSnr()
train(model=dccrn, optimizer=optimizer, criterion=criterion, train_iter=train_dataloader, test_iter=test_dataloader,
      max_epoch=40, device=para.device, batch_size=para.batch_size, log_path=save_file, just_test=False)
