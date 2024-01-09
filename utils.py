'''
description:the python coding file has many tools function. incluing :
data partition function、WarpSampler function、evaluate function.
'''
import sys
import copy
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue

def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):

    def sample():
        user = np.random.randint(1, usernum + 1)
        while len(user_train[0][user]) <= 1:
            user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)

        con_seq = np.zeros([maxlen], dtype=np.int32)
        pos_seq = np.zeros([maxlen], dtype=np.int32)
        neg_seq = np.zeros([maxlen], dtype=np.int32)
        time_seq = np.zeros([maxlen], dtype=np.int64)

        nxt = user_train[0][user][-1]
        idx = maxlen - 1

        ts = set(user_train[0][user])
        for i in range(maxlen):
            if 2+i < len(user_train[0][user]):
                seq[idx] = user_train[0][user][-2-i]
                con_seq[idx] = user_train[1][user][-2-i]
                pos_seq[idx] = user_train[2][user][-2-i]
                neg_seq[idx] = user_train[3][user][-2-i]
                time_seq[idx] = user_train[4][user][-2-i]

            else:
                seq[idx] = 0
                con_seq[idx] = 0
                pos_seq[idx] = 0
                neg_seq[idx] = 0
                time_seq[idx] = 0

            pos[idx] = nxt
            if nxt != 0:
                neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = seq[idx]
            idx -= 1
            if idx == -1 or i == maxlen-2:
                break

        return (user, seq, pos, neg, con_seq, pos_seq, neg_seq, time_seq)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())
        result_queue.put(zip(*one_batch))

def sample_function_English(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):

    def sample():
        user = np.random.randint(1, usernum + 1)
        while len(user_train[0][user]) <= 1:
            user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)

        neu_seq = np.zeros([maxlen], dtype=np.int32)
        pos_seq = np.zeros([maxlen], dtype=np.int32)
        neg_seq = np.zeros([maxlen], dtype=np.int32)
        vpos_seq = np.zeros([maxlen], dtype=np.int32)
        vneg_seq = np.zeros([maxlen], dtype=np.int32)
        time_seq = np.zeros([maxlen], dtype=np.int64)

        nxt = user_train[0][user][-1]
        idx = maxlen - 1

        ts = set(user_train[0][user])
        for i in range(maxlen):
            if 2+i < len(user_train[0][user]):
                seq[idx] = user_train[0][user][-2-i]
                vpos_seq[idx] = user_train[1][user][-2 - i]
                pos_seq[idx] = user_train[2][user][-2 - i]
                neu_seq[idx] = user_train[3][user][-2 - i]
                neg_seq[idx] = user_train[4][user][-2 - i]
                vneg_seq[idx] = user_train[5][user][-2 - i]
                time_seq[idx] = user_train[6][user][-2 - i]

            else:
                seq[idx] = 0
                neu_seq[idx] = 0
                pos_seq[idx] = 0
                neg_seq[idx] = 0
                vpos_seq[idx] = 0
                vneg_seq[idx] = 0
                time_seq[idx] = 0

            pos[idx] = nxt
            if nxt != 0:
                neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = seq[idx]
            idx -= 1
            if idx == -1 or i == maxlen-2:
                break

        return (user, seq, pos, neg, vpos_seq, pos_seq, neu_seq, neg_seq, vneg_seq, time_seq)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())
        result_queue.put(zip(*one_batch))

class WarpSampler(object):

    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=3, seed=np.random.randint(2e9)):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []

        for i in range(n_workers):
            self.processors.append(Process(target=sample_function,args=(User, usernum, itemnum, batch_size, maxlen,
                                                                        self.result_queue, seed)))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()

class WarpSampler_English(object):

    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=3, seed=np.random.randint(2e9)):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []

        for i in range(n_workers):
            self.processors.append(Process(target=sample_function_English,
                                           args=(User, usernum, itemnum, batch_size, maxlen, self.result_queue, seed)))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()

def ASSR_data_partition(fname):
    '''
    :param fname: 原数据集文件
    :return: 切分之后的包含原始情感的数据集的文件列表
    '''
    usernum = 0
    itemnum = 0

    User = defaultdict(list)  # 默认字典列表，当字典的键不存在时，返回[]
    Pos = defaultdict(list)
    Neg = defaultdict(list)
    Con = defaultdict(list)
    # 时间
    Time = defaultdict(list)

    # 字典
    user_train = {}
    user_valid = {}
    user_test = {}

    pos_train = {}
    pos_valid = {}
    pos_test = {}

    neg_train = {}
    neg_valid = {}
    neg_test = {}

    con_train = {}
    con_valid = {}
    con_test = {}

    time_train = {}
    time_valid = {}
    time_test = {}

    # assume user/item index starting from 1
    f = open('../../data/%s.rating' % fname, 'r')  # 打开data/bibi_new2(2).txt文件

    for line in f:
        u, i, rating, time, con, pos, neg = line.rstrip().split(',')  # 读取每一行的u,i的标志，中间以','隔开
        # 读取u,i的值
        u = int(u)
        i = int(i)
        # 获取用户数量和物品数量，但是注意，他们必须是按照顺序进行排列的
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)  # 把用户u的项目i添加到字典中
        Pos[u].append(pos)
        Neg[u].append(neg)
        Con[u].append(con)

        Time[u].append(time)

    for user in User:
        # 提取每个user的记录，然后计算它的个数，也就是每个user与item的交互次数
        nfeedback = len(User[user])
        if nfeedback < 3:
            # 当用户的交互记录小于3的时候，其全部放入训练集
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []

            pos_train[user] = Pos[user]
            pos_valid[user] = []
            pos_test[user] = []

            neg_train[user] = Neg[user]
            neg_valid[user] = []
            neg_test[user] = []

            con_train[user] = Con[user]
            con_valid[user] = []
            con_test[user] = []

            time_train[user] = Time[user]
            time_valid[user] = []
            time_test[user] = []


        else:
            # 当交互记录大于3的时候，除了倒数第一个和第二个以外，其余的全部放入训练集
            # 倒数第二个放入有效验证集，倒数第一个放入测试集
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])

            pos_train[user] = Pos[user][:-2]
            pos_valid[user] = []
            pos_valid[user].append(Pos[user][-2])
            pos_test[user] = []
            pos_test[user].append(Pos[user][-1])

            neg_train[user] = Neg[user][:-2]
            neg_valid[user] = []
            neg_valid[user].append(Neg[user][-2])
            neg_test[user] = []
            neg_test[user].append(Neg[user][-1])

            con_train[user] = Con[user][:-2]
            con_valid[user] = []
            con_valid[user].append(Con[user][-2])
            con_test[user] = []
            con_test[user].append(Con[user][-1])

            time_train[user] = Time[user][:-2]
            time_valid[user] = []
            time_valid[user].append(Time[user][-2])
            time_test[user] = []
            time_test[user].append(Time[user][-1])

    return [user_train, user_valid, user_test,
            con_train, con_valid, con_test,
            pos_train, pos_valid, pos_test,
            neg_train, neg_valid, neg_test,
            time_train, time_valid, time_test,
            usernum, itemnum]

def ASSR_data_partition_English(fname):
    '''
    :param fname: 原数据集文件
    :return: 切分之后的包含原始情感的数据集的文件列表
    '''
    usernum = 0
    itemnum = 0

    User = defaultdict(list)  # 默认字典列表，当字典的键不存在时，返回[]
    Pos = defaultdict(list)
    Neg = defaultdict(list)
    vPos = defaultdict(list)
    vNeg = defaultdict(list)
    Neu = defaultdict(list)
    Time = defaultdict(list)

    # 字典
    user_train = {}
    user_valid = {}
    user_test = {}

    pos_train = {}
    pos_valid = {}
    pos_test = {}

    vpos_train = {}
    vpos_valid = {}
    vpos_test = {}

    neg_train = {}
    neg_valid = {}
    neg_test = {}

    vneg_train = {}
    vneg_valid = {}
    vneg_test = {}

    neu_train = {}
    neu_valid = {}
    neu_test = {}

    time_train = {}
    time_valid = {}
    time_test = {}

    # assume user/item index starting from 1
    f = open('../../data/%s.rating' % fname, 'r')  # 打开data/bibi_new2(2).txt文件

    for line in f:
        u, i, rating, time, v_pos, pos, neu, neg, v_neg = line.rstrip().split('\t')  # 读取每一行的u,i的标志，中间以','隔开
        # 读取u,i的值
        u = int(u)
        i = int(i)
        # 获取用户数量和物品数量，但是注意，他们必须是按照顺序进行排列的
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)  # 把用户u的项目i添加到字典中
        Pos[u].append(pos)
        Neg[u].append(neg)
        vPos[u].append(v_pos)
        vNeg[u].append(v_neg)
        Neu[u].append(neu)
        Time[u].append(time)

    for user in User:
        # 提取每个user的记录，然后计算它的个数，也就是每个user与item的交互次数
        nfeedback = len(User[user])
        if nfeedback < 3:
            # 当用户的交互记录小于3的时候，其全部放入训练集
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []

            pos_train[user] = Pos[user]
            pos_valid[user] = []
            pos_test[user] = []

            vpos_train[user] = vPos[user]
            vpos_valid[user] = []
            vpos_test[user] = []

            neg_train[user] = Neg[user]
            neg_valid[user] = []
            neg_test[user] = []

            vneg_train[user] = vNeg[user]
            vneg_valid[user] = []
            vneg_test[user] = []

            neu_train[user] = Neu[user]
            neu_valid[user] = []
            neu_test[user] = []

            time_train[user] = Time[user]
            time_valid[user] = []
            time_test[user] = []


        else:
            # 当交互记录大于3的时候，除了倒数第一个和第二个以外，其余的全部放入训练集
            # 倒数第二个放入有效验证集，倒数第一个放入测试集
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])

            pos_train[user] = Pos[user][:-2]
            pos_valid[user] = []
            pos_valid[user].append(Pos[user][-2])
            pos_test[user] = []
            pos_test[user].append(Pos[user][-1])

            vpos_train[user] = vPos[user][:-2]
            vpos_valid[user] = []
            vpos_valid[user].append(vPos[user][-2])
            vpos_test[user] = []
            vpos_test[user].append(vPos[user][-1])

            neg_train[user] = Neg[user][:-2]
            neg_valid[user] = []
            neg_valid[user].append(Neg[user][-2])
            neg_test[user] = []
            neg_test[user].append(Neg[user][-1])

            vneg_train[user] = vNeg[user][:-2]
            vneg_valid[user] = []
            vneg_valid[user].append(vNeg[user][-2])
            vneg_test[user] = []
            vneg_test[user].append(vNeg[user][-1])

            neu_train[user] = Neu[user][:-2]
            neu_valid[user] = []
            neu_valid[user].append(Neu[user][-2])
            neu_test[user] = []
            neu_test[user].append(Neu[user][-1])

            time_train[user] = Time[user][:-2]
            time_valid[user] = []
            time_valid[user].append(Time[user][-2])
            time_test[user] = []
            time_test[user].append(Time[user][-1])

    return [user_train, user_valid, user_test,
            vpos_train, vpos_valid, vpos_test,
            pos_train, pos_valid, pos_test,
            neu_train, neu_valid, neu_test,
            neg_train, neg_valid, neg_test,
            vneg_train, vneg_valid, vneg_test,
            time_train, time_valid, time_test,
            usernum, itemnum]

def ASSR_evaluate_test(model, dataset, args):
    [train, valid, test,
     con_train, con_valid, con_test,
     pos_train, pos_valid, pos_test,
     neg_train, neg_valid, neg_test,
     time_train, time_valid, time_test,
     usernum, itemnum] = copy.deepcopy(dataset)

    # train：整条序列去掉最后两个，valid：倒数第二个，test：序列中最后一个，usernum：用户数量。itemnum：item总数
    NDCG_10 = 0.0
    HT_10 = 0.0
    NDCG_20 = 0.0
    HT_20 = 0.0

    valid_user = 0.0
    if usernum > 10000:           # 如果用户数量太对则对其采样
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)       # 生成用户序列

    for u in users:                         # 遍历每个用户
        if len(train[u]) < 1 or len(test[u]) < 1:
            continue          # 如果该用户的训练序列长度小于1，或者没有test的数据则跳过

        seq = np.zeros([args.maxlen], dtype=np.int32)               # 生成一个session——len长度的全0向量

        con_seq = np.zeros([args.maxlen], dtype=np.int32)
        pos_seq = np.zeros([args.maxlen], dtype=np.int32)
        neg_seq = np.zeros([args.maxlen], dtype=np.int32)
        time_seq = np.zeros([args.maxlen], dtype=np.int32)

        idx = args.maxlen - 1                   # 得到最后一位的索引
        seq[idx] = valid[u][0]                  # 最后一位维vaild数据
        idx -= 1
        for i in reversed(train[u]):            # 反向遍历训练session，将其填入seq中
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break

        idx = args.maxlen - 1
        con_seq[idx] = con_valid[u][0]
        idx -= 1
        for i in reversed(con_train[u]):
            con_seq[idx] = i
            idx -= 1
            if idx == -1:
                break


        idx = args.maxlen - 1
        pos_seq[idx] = pos_valid[u][0]
        idx -= 1
        for i in reversed(pos_train[u]):
            pos_seq[idx] = i
            idx -= 1
            if idx == -1:
                break



        idx = args.maxlen - 1
        neg_seq[idx] = neg_valid[u][0]
        idx -= 1
        for i in reversed(neg_train[u]):
            neg_seq[idx] = i
            idx -= 1
            if idx == -1:
                break

        idx = args.maxlen - 1
        time_seq[idx] = time_valid[u][0]
        idx -= 1
        for i in reversed(time_train[u]):
            time_seq[idx] = i
            idx -= 1
            if idx == -1:
                break

        rated = set(train[u])                   # 去除train——session中重复的item
        rated.add(0)                            # 加入0
        item_idx = [test[u][0]]                 # 得到test的目标item
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)           # 任意抽取一个item
            while t in rated:
                t = np.random.randint(1, itemnum + 1)       # 如果item存在session中则重新抽取一个，知道不存在session中为止
            item_idx.append(t)                              # 添加100个不存在session中的item，加上test共101

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], [con_seq], [pos_seq], [neg_seq], [time_seq], item_idx]])  # [1,101]
        predictions = predictions[0]  # [101] # - for 1st argsort DESC
        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1  # 记录一共验证了多少个用户

        if rank < 20:
            NDCG_20 += 1 / np.log2(rank + 2)
            HT_20 += 1
        if rank < 10:
            NDCG_10 += 1 / np.log2(rank + 2)
            HT_10 += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG_10 / valid_user, HT_10 / valid_user, NDCG_20 / valid_user, HT_20 / valid_user

def ASSR_evaluate_test_English(model, dataset, args):
    [train, valid, test,
     vpos_train, vpos_valid, vpos_test,
     pos_train, pos_valid, pos_test,
     neu_train, neu_valid, neu_test,
     neg_train, neg_valid, neg_test,
     vneg_train, vneg_valid, vneg_test,
     time_train, time_valid, time_test,
     usernum, itemnum] = copy.deepcopy(dataset)

    # train：整条序列去掉最后两个，valid：倒数第二个，test：序列中最后一个，usernum：用户数量。itemnum：item总数
    NDCG_10 = 0.0
    HT_10 = 0.0
    NDCG_20 = 0.0
    HT_20 = 0.0

    valid_user = 0.0
    if usernum > 10000:           # 如果用户数量太对则对其采样
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)       # 生成用户序列

    for u in users:                         # 遍历每个用户
        if len(train[u]) < 1 or len(test[u]) < 1:
            continue          # 如果该用户的训练序列长度小于1，或者没有test的数据则跳过

        seq = np.zeros([args.maxlen], dtype=np.int32)               # 生成一个session——len长度的全0向量

        neu_seq = np.zeros([args.maxlen], dtype=np.int32)
        pos_seq = np.zeros([args.maxlen], dtype=np.int32)
        neg_seq = np.zeros([args.maxlen], dtype=np.int32)
        vpos_seq = np.zeros([args.maxlen], dtype=np.int32)
        vneg_seq = np.zeros([args.maxlen], dtype=np.int32)
        time_seq = np.zeros([args.maxlen], dtype=np.int32)

        idx = args.maxlen - 1                   # 得到最后一位的索引
        seq[idx] = valid[u][0]                  # 最后一位维vaild数据
        idx -= 1
        for i in reversed(train[u]):            # 反向遍历训练session，将其填入seq中
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break

        idx = args.maxlen - 1
        neu_seq[idx] = neu_valid[u][0]
        idx -= 1
        for i in reversed(neu_train[u]):
            neu_seq[idx] = i
            idx -= 1
            if idx == -1:
                break

        idx = args.maxlen - 1
        pos_seq[idx] = pos_valid[u][0]
        idx -= 1
        for i in reversed(pos_train[u]):
            pos_seq[idx] = i
            idx -= 1
            if idx == -1:
                break

        idx = args.maxlen - 1
        vpos_seq[idx] = vpos_valid[u][0]
        idx -= 1
        for i in reversed(vpos_train[u]):
            vpos_seq[idx] = i
            idx -= 1
            if idx == -1:
                break

        idx = args.maxlen - 1
        neg_seq[idx] = neg_valid[u][0]
        idx -= 1
        for i in reversed(neg_train[u]):
            neg_seq[idx] = i
            idx -= 1
            if idx == -1:
                break

        idx = args.maxlen - 1
        vneg_seq[idx] = vneg_valid[u][0]
        idx -= 1
        for i in reversed(vneg_train[u]):
            vneg_seq[idx] = i
            idx -= 1
            if idx == -1:
                break

        idx = args.maxlen - 1
        time_seq[idx] = time_valid[u][0]
        idx -= 1
        for i in reversed(time_train[u]):
            time_seq[idx] = i
            idx -= 1
            if idx == -1:
                break

        rated = set(train[u])                   # 去除train——session中重复的item
        rated.add(0)                            # 加入0
        item_idx = [test[u][0]]                 # 得到test的目标item
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)           # 任意抽取一个item
            while t in rated:
                t = np.random.randint(1, itemnum + 1)       # 如果item存在session中则重新抽取一个，知道不存在session中为止
            item_idx.append(t)                              # 添加100个不存在session中的item，加上test共101

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], [vpos_seq], [pos_seq], [neu_seq],
                                                             [neg_seq], [vneg_seq], [time_seq], item_idx]])  # [1,101]
        predictions = predictions[0]  # [101] # - for 1st argsort DESC
        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1  # 记录一共验证了多少个用户

        if rank < 20:
            NDCG_20 += 1 / np.log2(rank + 2)
            HT_20 += 1
        if rank < 10:
            NDCG_10 += 1 / np.log2(rank + 2)
            HT_10 += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG_10 / valid_user, HT_10 / valid_user, NDCG_20 / valid_user, HT_20 / valid_user

# 评估衰减情感的ASSR-valid
def ASSR_evaluate_valid(model, dataset, args):
    [train, valid, test,
     con_train, con_valid, con_test,
     pos_train, pos_valid, pos_test,
     neg_train, neg_valid, neg_test,
     time_train, time_valid, time_test,
     usernum, itemnum] = copy.deepcopy(dataset)

    NDCG_10 = 0.0
    HT_10 = 0.0
    NDCG_20 = 0.0
    HT_20 = 0.0
    valid_user = 0.0
    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        # 新增
        con_seq = np.zeros([args.maxlen], dtype=np.int32)
        pos_seq = np.zeros([args.maxlen], dtype=np.int32)
        neg_seq = np.zeros([args.maxlen], dtype=np.int32)
        time_seq = np.zeros([args.maxlen], dtype=np.int64)

        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break

        idx = args.maxlen - 1
        for i in reversed(con_train[u]):
            con_seq[idx] = i
            idx -= 1
            if idx == -1:
                break

        idx = args.maxlen - 1
        for i in reversed(pos_train[u]):
            pos_seq[idx] = i
            idx -= 1
            if idx == -1:
                break

        idx = args.maxlen - 1
        for i in reversed(neg_train[u]):
            neg_seq[idx] = i
            idx -= 1
            if idx == -1:
                break

        idx = args.maxlen - 1
        for i in reversed(time_train[u]):
            time_seq[idx] = i
            idx -= 1
            if idx == -1:
                break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated:
                t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], [con_seq], [pos_seq], [neg_seq], [time_seq], item_idx]])
        predictions = predictions[0]
        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 20:
            NDCG_20 += 1 / np.log2(rank + 2)
            HT_20 += 1
        if rank < 10:
            NDCG_10 += 1 / np.log2(rank + 2)
            HT_10 += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG_10 / valid_user, HT_10 / valid_user, NDCG_20 / valid_user, HT_20 / valid_user

def ASSR_evaluate_valid_English(model, dataset, args):
    [train, valid, test,
     vpos_train, vpos_valid, vpos_test,
     pos_train, pos_valid, pos_test,
     neu_train, neu_valid, neu_test,
     neg_train, neg_valid, neg_test,
     vneg_train, vneg_valid, vneg_test,
     time_train, time_valid, time_test,
     usernum, itemnum] = copy.deepcopy(dataset)

    NDCG_10 = 0.0
    HT_10 = 0.0
    NDCG_20 = 0.0
    HT_20 = 0.0
    valid_user = 0.0
    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        # 新增
        neu_seq = np.zeros([args.maxlen], dtype=np.int32)
        pos_seq = np.zeros([args.maxlen], dtype=np.int32)
        neg_seq = np.zeros([args.maxlen], dtype=np.int32)
        vpos_seq = np.zeros([args.maxlen], dtype=np.int32)
        vneg_seq = np.zeros([args.maxlen], dtype=np.int32)
        time_seq = np.zeros([args.maxlen], dtype=np.int64)

        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break

        idx = args.maxlen - 1
        for i in reversed(neu_train[u]):
            neu_seq[idx] = i
            idx -= 1
            if idx == -1:
                break

        idx = args.maxlen - 1
        for i in reversed(pos_train[u]):
            pos_seq[idx] = i
            idx -= 1
            if idx == -1:
                break

        idx = args.maxlen - 1
        for i in reversed(vpos_train[u]):
            vpos_seq[idx] = i
            idx -= 1
            if idx == -1:
                break

        idx = args.maxlen - 1
        for i in reversed(neg_train[u]):
            neg_seq[idx] = i
            idx -= 1
            if idx == -1:
                break

        idx = args.maxlen - 1
        for i in reversed(vneg_train[u]):
            vneg_seq[idx] = i
            idx -= 1
            if idx == -1:
                break

        idx = args.maxlen - 1
        for i in reversed(time_train[u]):
            time_seq[idx] = i
            idx -= 1
            if idx == -1:
                break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated:
                t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], [vpos_seq], [pos_seq], [neu_seq],
                                                             [neg_seq], [vneg_seq], [time_seq], item_idx]])
        predictions = predictions[0]
        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 20:
            NDCG_20 += 1 / np.log2(rank + 2)
            HT_20 += 1
        if rank < 10:
            NDCG_10 += 1 / np.log2(rank + 2)
            HT_10 += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG_10 / valid_user, HT_10 / valid_user, NDCG_20 / valid_user, HT_20 / valid_user