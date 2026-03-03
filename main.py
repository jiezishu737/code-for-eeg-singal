import argparse
import logging
import os
import matplotlib.pyplot as plt  # 新增导入

from tqdm import tqdm
from my_model_new import Eeg_csp
from utils import Setup

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from dotmap import DotMap
from myutils import *

import torch
import torch.nn as nn
from torch.optim import Adam

result_logger = logging.getLogger('result')
result_logger.setLevel(logging.INFO)

config = {}


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initiate(args, train_loader, valid_loader, test_loader, subject):
    model = Eeg_csp(config)

    print(f"The model has {count_parameters(model):,} trainable parameters.")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(params=model.parameters(), lr=0.0005, weight_decay=3e-4)

    model = model.cuda()
    criterion = criterion.cuda()

    settings = {'model': model,
                'optimizer': optimizer,
                'criterion': criterion}

    return train_model(settings, args, train_loader, valid_loader, test_loader, subject)


def plot_training_curves(train_losses, train_accs, val_losses, val_accs, subject, dataset, time_len):
    """绘制训练曲线"""
    epochs = range(1, len(train_losses) + 1)

    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # 绘制损失曲线
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title(f'Loss Curves - {dataset} {subject} {time_len}s')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 绘制准确率曲线
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title(f'Accuracy Curves - {dataset} {subject} {time_len}s')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 调整布局
    plt.tight_layout()

    # 保存图片
    os.makedirs('training_plots', exist_ok=True)
    plot_path = f'/home/mydata/eeg/train_plots/{dataset}_{subject}_{time_len}s_{epochs}training_curve.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Training curve saved to: {plot_path}")


def train_model(settings, args, train_loader, valid_loader, test_loader, subject):
    model = settings['model']
    optimizer = settings['optimizer']
    criterion = settings['criterion']

    # 新增：记录训练过程的列表
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    def train(model, optimizer, criterion):
        model.train()
        train_acc_sum = 0
        train_loss_sum = 0
        batch_size = train_loader.batch_size

        for i_batch, batch_data in enumerate(train_loader):
            train_data, train_label = batch_data
            train_label = train_label.squeeze(-1)
            train_data, train_label = train_data.cuda(), train_label.cuda()
            preds,attn_dict = model(train_data)

            loss = criterion(preds, train_label.long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                train_loss_sum += loss.item() * batch_size
                predicted = preds.data.max(1)[1]
                train_acc_sum += predicted.eq(train_label).cpu().sum()

        return train_loss_sum / len(train_loader.dataset), train_acc_sum / len(train_loader.dataset)

    def evaluate(model, criterion, test=False):
        model.eval()
        if test:
            loader = test_loader
            num_batches = len(test_loader)
        else:
            loader = valid_loader
            num_batches = len(valid_loader)

        total_loss = 0.0
        test_acc_sum = 0
        batch_size = loader.batch_size

        with torch.no_grad():
            for i_batch, batch_data in enumerate(loader):
                test_data, test_label = batch_data
                test_label = test_label.squeeze(-1)
                test_data, test_label = test_data.cuda(), test_label.cuda()
                preds, attn_dict = model(test_data)

                total_loss += criterion(preds, test_label.long()).item() * batch_size
                preds = preds.detach()
                predicted = preds.data.max(1)[1]
                test_acc_sum += predicted.eq(test_label).cpu().sum()

        avg_loss = total_loss / (num_batches * batch_size)
        avg_acc = test_acc_sum / (num_batches * batch_size)

        return avg_loss, avg_acc

    # def evaluate_heatmap(model, criterion, test=False):
    #     model.eval()
    #     if test:
    #         loader = test_loader
    #         num_batches = len(test_loader)
    #     else:
    #         loader = valid_loader
    #         num_batches = len(valid_loader)
    #
    #     total_loss = 0.0
    #     test_acc_sum = 0
    #     batch_size = loader.batch_size
    #
    #     with torch.no_grad():
    #         for i_batch, batch_data in enumerate(loader):
    #             test_data, test_label = batch_data
    #             test_label = test_label.squeeze(-1)
    #             test_data, test_label = test_data.cuda(), test_label.cuda()
    #             preds, attn_dict = model(test_data)
    #
    #             attn_11 = attn_dict['dcsam_attn1']['attn_11']  # [batch, num_heads, S/2, S/2]
    #             # 取第一个样本、所有头的平均
    #             att_cross = attn_11[0].mean(dim=0).cpu().numpy()  # [S/2, S/2]
    #
    #             #att_cross = attn_dict['dcsam_attn1'][0, 0].cpu().numpy()  # 取 batch 0, head 0
    #
    #             plt.figure(figsize=(6, 5))
    #             plt.imshow(att_cross, cmap='hot', interpolation='nearest')
    #             plt.colorbar()
    #             plt.title('Cross-Attention (X2 attends to X1)')
    #             plt.xlabel('X1 time steps')
    #             plt.ylabel('X2 time steps')
    #             plt.savefig('/home/mydata/eeg/heatmap/dcsam_cross_attn.png', dpi=300)
    #             plt.close()
    #
    #
    #             # # DFFAM 通道注意力（取平均通道权重）
    #             # #ch_att = attn_dict['dffam_channel'][0, :, 0].cpu().numpy()  # [C]
    #             # ch_att = attn_dict['channel_att2'].mean(dim=0).cpu().numpy()
    #             # # 强制转换为至少一维数组
    #             # ch_att = np.atleast_1d(ch_att)  # 如果 ch_att 是标量，变为 [ch_att]
    #             #
    #             # # 绘制 top-10 通道条形图
    #             # top_k = min(10, len(ch_att))
    #             # #top_k = 10
    #             # indices = np.argsort(ch_att)[-top_k:]
    #             # plt.barh(range(top_k), ch_att[indices])
    #             # plt.yticks(range(top_k), [f'Ch{i}' for i in indices])
    #             # plt.xlabel('Attention weight')
    #             # plt.title('Top-10 EEG Channels by DFFAM Attention')
    #             # plt.tight_layout()
    #             # plt.savefig('/home/mydata/eeg/heatmap/dffam_channel_top10.png', dpi=300)
    #             # plt.close()
    #
    #             total_loss += criterion(preds, test_label.long()).item() * batch_size
    #             preds = preds.detach()
    #             predicted = preds.data.max(1)[1]
    #             test_acc_sum += predicted.eq(test_label).cpu().sum()
    #
    #     avg_loss = total_loss / (num_batches * batch_size)
    #     avg_acc = test_acc_sum / (num_batches * batch_size)
    #
    #     return avg_loss, avg_acc

    def evaluate_heatmap(model, criterion, test=False, subject=None, dataset=None, time_len=None):
        model.eval()
        if test:
            loader = test_loader
            num_batches = len(test_loader)
        else:
            loader = valid_loader
            num_batches = len(valid_loader)

        total_loss = 0.0
        test_acc_sum = 0
        batch_size = loader.batch_size

        # 累加器初始化
        channel_att1_sum = None
        channel_att2_sum = None
        pos_att1_sum = None
        pos_att2_sum = None
        attn_11_sum = None
        attn_22_sum = None
        attn_12_sum = None
        attn_21_sum = None
        num_samples = 0

        with torch.no_grad():
            for i_batch, batch_data in enumerate(loader):
                test_data, test_label = batch_data
                test_label = test_label.squeeze(-1)
                test_data, test_label = test_data.cuda(), test_label.cuda()
                preds, attn_dict = model(test_data)

                # 累加注意力权重（移至CPU）
                if channel_att1_sum is None:
                    channel_att1_sum = torch.zeros_like(attn_dict['channel_att1']).cpu()
                    channel_att2_sum = torch.zeros_like(attn_dict['channel_att2']).cpu()
                    pos_att1_sum = torch.zeros_like(attn_dict['pos_att1']).cpu()
                    pos_att2_sum = torch.zeros_like(attn_dict['pos_att2']).cpu()
                    attn_11_sum = torch.zeros_like(attn_dict['dcsam_attn1']['attn_11']).cpu()
                    attn_22_sum = torch.zeros_like(attn_dict['dcsam_attn1']['attn_22']).cpu()
                    attn_12_sum = torch.zeros_like(attn_dict['dcsam_attn1']['attn_12']).cpu()
                    attn_21_sum = torch.zeros_like(attn_dict['dcsam_attn1']['attn_21']).cpu()

                channel_att1_sum += attn_dict['channel_att1'].cpu()
                channel_att2_sum += attn_dict['channel_att2'].cpu()
                pos_att1_sum += attn_dict['pos_att1'].cpu()
                pos_att2_sum += attn_dict['pos_att2'].cpu()
                attn_11_sum += attn_dict['dcsam_attn1']['attn_11'].cpu()
                attn_22_sum += attn_dict['dcsam_attn1']['attn_22'].cpu()
                attn_12_sum += attn_dict['dcsam_attn1']['attn_12'].cpu()
                attn_21_sum += attn_dict['dcsam_attn1']['attn_21'].cpu()
                num_samples += test_data.size(0)

                total_loss += criterion(preds, test_label.long()).item() * batch_size
                preds = preds.detach()
                predicted = preds.data.max(1)[1]
                test_acc_sum += predicted.eq(test_label).cpu().sum()

        avg_loss = total_loss / (num_batches * batch_size)
        avg_acc = test_acc_sum / (num_batches * batch_size)

        # 计算平均注意力
        channel_att1_avg = channel_att1_sum.mean(dim=0).numpy()  # 形状 [c]
        channel_att2_avg = channel_att2_sum.mean(dim=0).numpy()
        pos_att1_avg = pos_att1_sum.mean(dim=0).numpy()  # 形状 [s]
        pos_att2_avg = pos_att2_sum.mean(dim=0).numpy()
        attn_11_avg = attn_11_sum.mean(dim=(0, 1)).numpy()  # [S/2, S/2]
        attn_22_avg = attn_22_sum.mean(dim=(0, 1)).numpy()
        attn_12_avg = attn_12_sum.mean(dim=(0, 1)).numpy()
        attn_21_avg = attn_21_sum.mean(dim=(0, 1)).numpy()

        # 创建保存目录
        save_dir = '/home/mydata/eeg/attention_plots/'
        os.makedirs(save_dir, exist_ok=True)

        # 绘制通道注意力条形图
        plt.figure(figsize=(10, 5))
        plt.bar(range(len(channel_att1_avg)), channel_att1_avg)
        plt.xlabel('Channel Index')
        plt.ylabel('Average Attention Weight')
        plt.title(f'Average Channel Attention (DFFAM1) - {dataset} {subject} {time_len}s')
        plt.savefig(os.path.join(save_dir, f'{dataset}_{subject}_{time_len}s_channel_att1.png'), dpi=300,
                    bbox_inches='tight')
        plt.close()

        # 类似绘制 channel_att2
        plt.figure(figsize=(10, 5))
        plt.bar(range(len(channel_att2_avg)), channel_att2_avg)
        plt.xlabel('Channel Index')
        plt.ylabel('Average Attention Weight')
        plt.title(f'Average Channel Attention (DFFAM2) - {dataset} {subject} {time_len}s')
        plt.savefig(os.path.join(save_dir, f'{dataset}_{subject}_{time_len}s_channel_att2.png'), dpi=300,
                    bbox_inches='tight')
        plt.close()

        # 绘制时间注意力折线图
        plt.figure(figsize=(12, 4))
        plt.plot(pos_att1_avg)
        plt.xlabel('Time Step')
        plt.ylabel('Average Attention Weight')
        plt.title(f'Average Temporal Attention (DFFAM1) - {dataset} {subject} {time_len}s')
        plt.savefig(os.path.join(save_dir, f'{dataset}_{subject}_{time_len}s_pos_att1.png'), dpi=300,
                    bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(12, 4))
        plt.plot(pos_att2_avg)
        plt.xlabel('Time Step')
        plt.ylabel('Average Attention Weight')
        plt.title(f'Average Temporal Attention (DFFAM2) - {dataset} {subject} {time_len}s')
        plt.savefig(os.path.join(save_dir, f'{dataset}_{subject}_{time_len}s_pos_att2.png'), dpi=300,
                    bbox_inches='tight')
        plt.close()

        # 绘制 DCSAM 注意力热图
        plt.figure(figsize=(6, 5))
        plt.imshow(attn_11_avg, cmap='hot', interpolation='nearest')
        plt.xticks(range(attn_11_avg.shape[1]))
        plt.yticks(range(attn_11_avg.shape[0]))
        plt.colorbar()
        plt.title(f'DCSAM Self-Attention (X1) - {dataset} {subject} {time_len}s')
        plt.xlabel('X1 Time Steps')
        plt.ylabel('X1 Time Steps')
        plt.savefig(os.path.join(save_dir, f'{dataset}_{subject}_{time_len}s_attn11.png'), dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(6, 5))
        plt.imshow(attn_22_avg, cmap='hot', interpolation='nearest')
        plt.xticks(range(attn_22_avg.shape[1]))
        plt.yticks(range(attn_22_avg.shape[0]))
        plt.colorbar()
        plt.title(f'DCSAM Self-Attention (X2) - {dataset} {subject} {time_len}s')
        plt.xlabel('X2 Time Steps')
        plt.ylabel('X2 Time Steps')
        plt.savefig(os.path.join(save_dir, f'{dataset}_{subject}_{time_len}s_attn22.png'), dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(6, 5))
        plt.imshow(attn_12_avg, cmap='hot', interpolation='nearest')
        plt.xticks(range(attn_12_avg.shape[1]))
        plt.yticks(range(attn_12_avg.shape[0]))
        plt.colorbar()
        plt.title(f'DCSAM Cross-Attention (X2 -> X1) - {dataset} {subject} {time_len}s')
        plt.xlabel('X1 Time Steps')
        plt.ylabel('X2 Time Steps')
        plt.savefig(os.path.join(save_dir, f'{dataset}_{subject}_{time_len}s_attn12.png'), dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(6, 5))
        plt.imshow(attn_21_avg, cmap='hot', interpolation='nearest')
        plt.xticks(range(attn_21_avg.shape[1]))
        plt.yticks(range(attn_21_avg.shape[0]))
        plt.colorbar()
        plt.title(f'DCSAM Cross-Attention (X1 -> X2) - {dataset} {subject} {time_len}s')
        plt.xlabel('X2 Time Steps')
        plt.ylabel('X1 Time Steps')
        plt.savefig(os.path.join(save_dir, f'{dataset}_{subject}_{time_len}s_attn21.png'), dpi=300, bbox_inches='tight')
        plt.close()

        return avg_loss, avg_acc

    epochs_without_improvement = 0
    best_epoch = 1
    best_valid = float('inf')

    # 修改训练循环，记录每个epoch的指标
    for epoch in tqdm(range(1, args.max_epoch + 1), desc='Training Epoch', leave=False):
        train_loss, train_acc = train(model, optimizer, criterion)
        val_loss, val_acc = evaluate(model, criterion, test=False)

        # 记录当前epoch的指标
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print()
        print(
            'Epoch {:2d} Finsh | Subject {} | Train Loss {:5.4f} | Train Acc {:5.4f} | Valid Loss {:5.4f} | Valid Acc '
            '{:5.4f}'.format(
                epoch,
                args.name,
                train_loss,
                train_acc,
                val_loss,
                val_acc))

        if val_loss < best_valid:
            best_valid = val_loss
            epochs_without_improvement = 0
            best_epoch = epoch
            print(f"Saved model at pre_trained_models/{save_load_name(args, name=args.name)}.pt!")
            save_model(args, model, name=args.name)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement > 10:
                print(f"Early stopping at epoch {epoch}")
                break

    # 绘制训练曲线
    plot_training_curves(train_losses, train_accs, val_losses, val_accs, subject, args.dataset, args.time_len)

    model = load_model(args, name=args.name)
    test_loss, test_acc = evaluate_heatmap(model, criterion, test=True, subject=subject, dataset=args.dataset, time_len=args.time_len)



    #heat map




    print(f'Best epoch: {best_epoch}')
    print(f"Subject: {subject}, Acc: {test_acc:.2f}")

    return test_loss, test_acc


def main(name="S1", time_len=2, dataset="DTU"):
    best_acc = float('-inf')

    args = DotMap()
    args.name = name
    args.max_epoch = 100
    args.time_len = time_len  # 新增参数
    args.dataset = dataset  # 新增参数

    train_loader, valid_loader, test_loader = getData(name, time_len, dataset)
    config['Data_shape'] = train_loader.dataset.data.shape
    print('Data shape:', config['Data_shape'])

    for i in range(10):
        i = i + 1
        print(f'Warning : This is epoch {i}')
        loss, acc = initiate(args, train_loader, valid_loader, test_loader, args.name)
        print(loss, acc.item())
        info_msg = f'{dataset}_{name}_{str(time_len)}s loss:{str(loss)} acc:{str(acc.item())}'
        if acc > best_acc:
            best_acc = acc

    info_msg = f'{dataset}_{name}_{str(time_len)}s loss:{str(loss)} best_acc:{str(best_acc)}'
    result_logger.info(info_msg)

    return loss, acc


if __name__ == "__main__":
    file_handler = logging.FileHandler('/home/mydata/eeg/log/result.log')
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    result_logger.addHandler(file_handler)

    main()