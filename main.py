import torch
from torch import nn
import csv
import matplotlib.pyplot as plt
import numpy as np
#数据处理
X_train_fpath = './data/X_train'
Y_train_fpath = './data/Y_train'
X_test_fpath = './data/X_test'

with open(X_train_fpath) as f:
    next(f)
    X_train = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)
with open(Y_train_fpath) as f:
    next(f)
    Y_train = np.array([line.strip('\n').split(',')[1] for line in f], dtype = float)
with open(X_test_fpath) as f:
    next(f)
    X_test = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)

def _train_dev_split(X, Y, dev_ratio = 0.25):
    # This function spilts data into training set and development set.
    train_size = int(len(X) * (1 - dev_ratio))
    return X[:train_size], Y[:train_size], X[train_size:], Y[train_size:]

def _normalize(X, train=True, specified_column=None, X_mean=None, X_std=None):
    if specified_column == None:
        specified_column = np.arange(X.shape[1])
    if train:
        X_mean = np.mean(X[:, specified_column], 0).reshape(1, -1)
        X_std = np.std(X[:, specified_column], 0).reshape(1, -1)

    X[:, specified_column] = (X[:, specified_column] - X_mean) / (X_std + 1e-8)
    return X, X_mean, X_std

dev_ratio = 0.1
X_train, Y_train, X_dev, Y_dev = _train_dev_split(X_train, Y_train, dev_ratio = dev_ratio)
X_train, X_mean, X_std =_normalize(X_train)
X_dev,_,_=_normalize(X_dev,X_mean=X_mean,X_std=X_std)
X_test,_,_=_normalize(X_test,X_mean=X_mean,X_std=X_std)

#构建模型
X_train=torch.from_numpy(X_train)
Y_train=torch.from_numpy(Y_train)
X_dev=torch.from_numpy(X_dev)
Y_dev=torch.from_numpy(Y_dev)
X_test=torch.from_numpy(X_test)
#完成numpy到tensor的转换

train_dataset=torch.utils.data.TensorDataset(X_train,Y_train)
dev_dataset=torch.utils.data.TensorDataset(X_dev,Y_dev)
test_dataset=torch.utils.data.TensorDataset(X_test)
#对给定的tensor数据(样本和标签)，将它们包装成dataset
train_loader=torch.utils.data.DataLoader(train_dataset,shuffle=True,batch_size=256)
dev_loader=torch.utils.data.DataLoader(dev_dataset,shuffle=True,batch_size=256)
test_loader=torch.utils.data.DataLoader(test_dataset,shuffle=True,batch_size=256)
#数据加载器，组合数据集和采样器，并在数据集上提供单进程或多进程迭代器。
#它可以对我们上面所说的数据集Dataset作进一步的设置。
'''
dataset (Dataset) – 加载数据的数据集。

batch_size (int, optional) – 每个batch加载多少个样本(默认: 1)。

shuffle (bool, optional) – 设置为True时会在每个epoch重新打乱数据(默
认: False).

sampler (Sampler, optional) – 定义从数据集中提取样本的策略。如果指
定，则shuffle必须设置成False。

num_workers (int, optional) – 用多少个子进程加载数据。0表示数据将在
主进程中加载(默认: 0)

pin_memory：内存寄存，默认为False。在数据返回前，是否将数据复制到
CUDA内存中。

drop_last (bool, optional) – 如果数据集大小不能被batch size整除，
则设置为True后可删除最后一个不完整的batch。如果设为False并且数据集的
大小不能被batch size整除，则最后一个batch将更小。(默认: False)

timeout：是用来设置数据读取的超时时间的，如果超过这个时间还没读取
到数据的话就会报错。 所以，数值必须大于等于0。
'''
print(X_train.shape,len(X_train[0]),Y_train.shape)

input_size=510
output_size=2
learning_rate=0.0001
epoches=15

model=torch.nn.Sequential(
    torch.nn.Linear(input_size,512),
    torch.nn.ReLU(),
    torch.nn.Linear(512,72),
    torch.nn.Dropout(0.5),
    torch.nn.ReLU(),
    torch.nn.Linear(72,output_size),
    torch.nn.ReLU(),
)
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)

training_loss_list=[]
dev_loss_list=[]
acc_list=[]
dev_acc_list=[]
max_acc=0
print("Training Start")
for epoch in range(epoches):
        running_loss=0
        dev_loss_total=0
        correct_total=0
        labels_total=0
        for i,data in enumerate(train_loader):
            inputs,labels=data
            #print(i,inputs,labels)
            optimizer.zero_grad()
            inputs=torch.tensor(inputs,dtype=torch.float32)
            outputs=model(inputs)
            loss=criterion(outputs,labels.long())
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()

            _, predict = torch.max(outputs, 1)
            #dim是max函数索引的维度0/1，0是每列的最大值，1是每行的最大值
            correct_total+=(predict == labels).sum().item()
            labels_total+=len(labels)

        acc = correct_total/labels_total
        acc_list.append(acc)
        training_loss_list.append(running_loss/labels_total)
        if epoch%1==0:
            # print(i)
            print("epoch",epoch,"loss={:.5}".format(running_loss/labels_total),"acc={:.5}".format(acc))
            dev_loss = 0
            dev_acc = 0
            dev_correct_total = 0
            dev_labels_total = 0
#一边训练一边验证
            with torch.no_grad():
                for data in dev_loader:
                    dev_inputs, dev_labels = data

                    dev_outputs = model(dev_inputs.float())
                    loss = criterion(dev_outputs, dev_labels.long())
                    dev_loss += loss.item()
                    _, dev_predict = torch.max(dev_outputs, 1)
                    dev_correct_total += (dev_predict == dev_labels).sum().item()
                    dev_labels_total += len(dev_labels)

                    dev_acc = dev_correct_total / dev_labels_total
                dev_loss_list.append(dev_loss/ dev_labels_total)
                dev_acc_list.append(dev_acc)
                print("[dev_loss]={:.5}".format(dev_loss / dev_labels_total), "[dev_acc]={:.5}".format(dev_acc))

            if dev_acc>max_acc:
                max_acc=dev_acc
                torch.save(model.state_dict(), 'test.pt')
                print("model saved,max_acc=",max_acc)
#选取在测试集中表现最好的模型，并保存

plt.plot(np.arange(epoches),training_loss_list)
plt.plot(np.arange(epoches),dev_loss_list)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
#损失值的图像
plt.plot(np.arange(epoches),acc_list)
plt.plot(np.arange(epoches),dev_acc_list)
plt.xlabel("Epoch")
plt.ylabel("Acc")
plt.show()
print("Finshed Training")
#精确度的图像

test_predict=model(X_test.float())
_,test_predict=torch.max(test_predict,1)
print("len=",len(test_predict),test_predict)

with open('classification_submit.csv', mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'label']
    csv_writer.writerow(header)
    for i in range(len(test_predict)):
        row = [ str(i), test_predict[i].item()]
        csv_writer.writerow(row)