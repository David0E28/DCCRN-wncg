# File       : eval.py
# Time       ：2023/7/10 19:13
# Author     ：David
# Description：github@https://github.com/David0E28
from pesq import pesq
import torch
import matplotlib.pyplot as plt
import wav_loader as loader
from hparams import hparams
from torch.utils.data import DataLoader
from train_utils import my_collect

para = hparams()

test_dataset = loader.WavDataset(para.file_scp_test, frame_dur=37.5)

PATH_model = "M:\DCCRN\logs\parameter_epoch8_2023-07-10 00-17-01.pth"

test_dataset = loader.WavDataset(para.file_scp_test, frame_dur=37.5)
test_dataloader = DataLoader(test_dataset, batch_size=para.load_batch, shuffle=True, collate_fn=my_collect)
def eval_pesq(PATH, test_iter, device, batch_size):
    model = torch.load(PATH)
    model.eval()
    with torch.no_grad():
        for ind, (x, y) in enumerate(test_iter):
            # x = x.view(x.size(0) * x.size(1), x.size(2)).to(device).float()
            # y = y.view(y.size(0) * y.size(1), y.size(2)).to(device).float()
            x = x.to(device).float()
            y = y.to(device).float()
            range_end = x.size(0) - (x.size(0) % batch_size) - batch_size - 1
            for index in range(0, range_end, batch_size):
                x_item = x[index:index + batch_size, :].squeeze(0)
                y_item = y[index:index + batch_size, :].squeeze(0)
                y_p = model(x_item, train=False)
                #cpu memory will not auto free!!!!!
                plt.plot(x_item.flatten().to("cpu").numpy())
                plt.plot(y_p.flatten().to("cpu").numpy())
                plt.show()
                print("增强前", pesq(16000, y_item.flatten().to("cpu").numpy(), x_item.flatten().to("cpu").numpy(),  "nb"))
                print('增强后', pesq(16000, y_item.flatten().to("cpu").numpy(), y_p.squeeze(1).flatten().to("cpu").numpy(),  "nb"))
                input("Press Enter to continue...")
if __name__ == "__main__":
    eval_pesq(PATH=PATH_model, test_iter=test_dataloader, device=para.device, batch_size=para.batch_size)