import os
import time
import argparse
from ASSR.utils import *
from model.ASSR.ASSR_model import ASSR
import torch

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', default='../../experiments/ASSR/')
    parser.add_argument('--dataset', default='bili_newdata', type=str)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=50, type=int)
    parser.add_argument('--hidden_units', default=20, type=int)
    parser.add_argument('--num_epochs', default=500, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_blocks', default=2, type=int)
    parser.add_argument('--num_heads', default=5, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--train_dir', default='default', type=str)
    parser.add_argument('--inference_only', default=False, type=str2bool)
    parser.add_argument('--state_dict_path', default=None, type=str)
    args = parser.parse_args()

    file_name = 'ASSR' + '.dataset={}.lr={}.maxlen={}.hidden={}.epoch={}.batch={}.layer={}.head={}.dropout={}'
    file_name = file_name.format(args.dataset, args.lr,args.maxlen, args.hidden_units, args.num_epochs,args.batch_size,
                                 args.num_blocks, args.num_heads, args.dropout_rate)

    if not os.path.isdir(args.result + '_' + args.train_dir):
        os.makedirs(args.result + '_' + args.train_dir)
    with open(os.path.join(args.result + '_' + args.train_dir, file_name+'_args.txt'), 'w') as f:
        f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
    f.close()

    # 完成数据集切分
    dataset = ASSR_data_partition(args.dataset)
    [user_train, user_valid, user_test,
     con_train, con_valid, con_test,
     pos_train, pos_valid, pos_test,
     neg_train, neg_valid, neg_test,
     time_train, time_valid, time_test,
     usernum, itemnum] = dataset

    num_batch = len(user_train) // args.batch_size
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))

    save_path = os.path.join(args.result, file_name + '_log.txt')
    f = open(save_path, 'w')
    seed = np.random.randint(2e9)
    sampler = WarpSampler([user_train, con_train, pos_train, neg_train, time_train],
                          usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=1, seed=seed)

    # 调用模型
    model = ASSR(usernum, itemnum, args).to(args.device)

    for name, param in model.named_parameters():        # 参数初始化
        try:
            torch.nn.init.xavier_uniform_(param.data)   # 使得初始参数服从均匀分布
        except:
            pass

    model.train()                                       # enable model training
    epoch_start_idx = 1
    bce_criterion = torch.nn.BCEWithLogitsLoss()        # torch.nn.BCELoss()：二进制交叉熵损失函数
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))  # 亚当优化器

    T = 0.0
    t0 = time.time()
    a = []
    # 存放训练结果
    best_vaile = {0: [0, 0, 0, 0]}
    best_test = {0: [0, 0, 0, 0]}
    all_data = []

    # 进入迭代训练
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        for step in range(num_batch):
            # 获取训练序列
            u, seq, pos, neg, con_seq, pos_seq, neg_seq, time_seq = sampler.next_batch()
            u, seq, pos, neg, con_seq, pos_seq, neg_seq, time_seq =\
                np.array(u), np.array(seq), np.array(pos), np.array(neg), np.array(con_seq), np.array(pos_seq), np.array(neg_seq), np.array(time_seq)

            pos_logits, neg_logits = model(u, seq, con_seq, pos_seq, neg_seq, time_seq, pos, neg)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape,
                                                                                                   device=args.device)

            adam_optimizer.zero_grad()                                      # 先将梯度归零
            indices = np.where(pos != 0)                                    # 得到正样不为0的索引
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])  # 正loss
            loss += bce_criterion(neg_logits[indices], neg_labels[indices]) # 加上负采样loss
            for param in model.item_emb.parameters():
                loss += args.l2_emb * torch.norm(param)
            loss.backward()
            adam_optimizer.step()
            print("loss in epoch {} iteration {}: {}".format(epoch, step,
                                                             loss.item()))  # expected 0.4~0.6 after init few epochs

        if epoch % 20 == 0:
            data_list = []
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating', end='')
            t_test = ASSR_evaluate_test(model, dataset, args)            # 评估测试集
            t_valid = ASSR_evaluate_valid(model, dataset, args)     # 评估验证集
            print('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)'
                  % (epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1]))

            if t_valid[1] > best_vaile[0][1]:
                best_vaile[0] = t_valid
            if t_test[1] > best_test[0][1]:
                best_test[0] = t_test

            # NDCG10\HIT10\NDCG20\HIT20
            data_list.append(t_test[0])
            data_list.append(t_test[1])
            data_list.append(t_test[2])
            data_list.append(t_test[3])
            all_data.append(data_list)

            # 写入文件
            # valid:NDCG10,HR10;test:NDCG10,HR10
            # f.write(t_test + '\n')
            f.write(str(t_test[0])+',')
            f.write(str(t_test[1])+',')
            f.write(str(t_test[2])+',')
            f.write(str(t_test[3])+'\n')
            f.flush()
            t0 = time.time()
            model.train()

        # 训练完成
        if epoch == args.num_epochs:
            folder = args.result + '_' + args.train_dir
            torch.save(model.state_dict(), os.path.join(folder, file_name + '.pth'))

    f.close()
    sampler.close()
    print('best valid:', best_vaile[0])
    print('best test:', best_test[0])
    print("Done")
