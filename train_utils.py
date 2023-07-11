import torch
import matplotlib.pyplot as plt
import librosa
import pickle
from pesq import pesq
import os
import time


def pesq_fn(y_truth, y_valid):
    # y_truth, y_valid = cap(y_truth, y_valid)

    y_truth = librosa.core.resample(y_truth, 16000, 16000)
    y_valid = librosa.core.resample(y_valid, 16000, 16000)
    return pesq(16000, y_truth, y_valid, "nb")


def test_epoch(model, test_iter, device, criterion, batch_size, test_all=False):
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        loss_sum = 0
        i = 0
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
                # if index % 3999 == 0:
                    # plt.plot(y_item.flatten().to("cpu").numpy())
                    # plt.plot(x_item.flatten().to("cpu").numpy())
                    # plt.show()
                    # print("增强前", pesq(16000, y_item.flatten().to("cpu").numpy(), x_item.flatten().to("cpu").numpy(),  "nb"))
                    # print('增强后', pesq(16000, y_item.flatten().to("cpu").numpy(), y_p.squeeze(1).flatten().to("cpu").numpy(),  "nb"))
                loss = criterion(source=y_item.unsqueeze(1), estimate_source=y_p).to(device)
                loss_sum += loss.item()
                i += 1
            if not test_all:
                break
    return loss_sum / i


def train(model, optimizer, criterion, train_iter, test_iter, max_epoch, device, batch_size, log_path, just_test=False):
    train_losses = []
    test_losses = []
    for epoch in range(max_epoch):
        loss_sum = 0
        i = 0
        for step, (x, y) in enumerate(train_iter):
            # x = x.view(x.size(0) * x.size(1), x.size(2)).to(device).float()
            # y = y.view(y.size(0) * y.size(1), y.size(2)).to(device).float()
            x = x.to(device).float()
            y = y.to(device).float()
            # shuffle = torch.randperm(x.size(0))
            # x = x[shuffle]
            # y = y[shuffle]
            range_end = x.size(0) - (x.size(0) % batch_size) - batch_size - 1
            for index in range(0, range_end, batch_size):

                model.train()

                x_item = x[index:index + batch_size, :].squeeze(0)
                y_item = y[index:index + batch_size, :].squeeze(0)
                optimizer.zero_grad()
                y_p = model(x_item)
                loss = criterion(source=y_item.unsqueeze(1), estimate_source=y_p).to(device)
                if step == 0 and index == 0 and epoch == 0:
                    loss.backward()
                    loss_sum += loss.item()
                    i += 1
                    test_loss = test_epoch(model, test_iter, device, criterion, batch_size=batch_size, test_all=False)
                    print(
                        "first test step:%d,ind:%d,train loss:%.5f,test loss:%.5f" % (
                            step, index, loss_sum / i, test_loss)
                    )
                    train_losses.append(loss_sum / i)
                    test_losses.append(test_loss)
                else:
                    loss.backward()
                    optimizer.step()
                    loss_sum += loss.item()
                    i += 1
            if  step == len(train_iter) - 1:
                test_loss = test_epoch(model, test_iter, device, criterion, batch_size=batch_size, test_all=False)
                print(
                    "epoch:%d,step:%d,train loss:%.5f,test loss:%.5f,time:%s" % (
                        epoch, step, loss_sum / i, test_loss, time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
                    )
                )
                train_losses.append(loss_sum / i)
                test_losses.append(test_loss)
                plt.plot(train_losses)
                plt.plot(test_losses)
                plt.savefig(os.path.join(log_path, "loss_time%s_epoch%d_step%d.png" % (
                    time.strftime("%Y-%m-%d %H-%M-%S", time.localtime()), epoch, step)), dpi=150)
                plt.show()
            if step == len(train_iter) - 1:
                print("save model,epoch:%d,step:%d,time:%s" % (
                    epoch, step, time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())))
                torch.save(model, os.path.join(log_path, "parameter_epoch%d_%s.pth" % (
                    epoch, time.strftime("%Y-%m-%d %H-%M-%S", time.localtime()))))
                pickle.dump({"train loss": train_losses, "test loss": test_losses},
                            open(os.path.join(log_path, "loss_time%s_epoch%d.log" % (
                                time.strftime("%Y-%m-%d %H-%M-%S", time.localtime()), epoch)), "wb"))
            if just_test:
                break


def my_collect(batch):
    batch_X = [item[0] for item in batch]
    batch_Y = [item[1] for item in batch]
    batch_X = torch.cat(batch_X, 0)
    batch_Y = torch.cat(batch_Y, 0)
    return[batch_X.float(), batch_Y.float()]

