import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR


transform = transforms.Compose(
    [transforms.Resize((224,224)),          # 由于挺多预训练模型是基于ImageNet,因此将其转换成224*224的尺寸
     transforms.ToTensor()])


# 设置Batch_size， 好让DataLoader切分，暂时先不分validation set，如果想分的话，可以从trainset.data索引出来。
# 坑：minst是.train_data, .train_label,而CIFAR是.data。而torchvision.datasets文档里也找不到这些属性。
batch_size=64
trainset = torchvision.datasets.CIFAR10(root='./data/cifar-10-python', train=True,
                                        download=False, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data/cifar-10-python', train=False,
                                       download=False, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False)

# model definition
model = 'EfficientNet-b0'

# 本次实验只探索了三种模型，ResNet50,EfficientNet-b0,DenseNet121
# 由于用的是Cifar10,十分类数据，因此将网络的输出层的输出节点改成10
if model == 'ResNet50':
    net = torchvision.models.resnet50(pretrained=True)
    net.fc.out_features=10
elif model == 'EfficientNet-b0':
    from efficientnet_pytorch import EfficientNet
    net = EfficientNet.from_pretrained('efficientnet-b0')
    net._fc.out_features = 10
elif model == 'DenseNet121':
    net = torchvision.models.densenet121(pretrained=True)
    net.classifier.out_features = 10

print(net)

# 训练配置
Epoch = 20
lr = 0.001  # 初始学习率
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
criterion = torch.nn.CrossEntropyLoss()

# 除了Adam优化器外，还使用余弦退火的学习率衰减策略
cos_lr = CosineAnnealingLR(optimizer, T_max=Epoch)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
net.to(device);

# 验证
def compute_val_loss(net, val_loader,  criterion, device):
    # 指定不计算梯度，若不指定pytorch的auto-grad会在验证时帮你爆显存
    with torch.no_grad():
        # 关闭训练时需要，但测试集不必要的策略，如dropout，BN等
        net.eval()
        num_of_batch = len(val_loader)
        losses = 0

        for i, data in enumerate(val_loader):
            x, y = data
            x, y = x.to(device), y.to(device)

            y_pred = net(x)
            val_loss = criterion(y_pred, y)
            losses += val_loss.item()

        val_loss = losses/num_of_batch
        return val_loss

print('Training...')
print('Model:', model)
from tqdm import tqdm
loss_curse=[]
val_loss_curse = []
lr_curse = []

count = 0
min_val_loss = float('inf')

import time
for epoch in tqdm(range(Epoch)):
    losses = 0
    for i, data in enumerate(train_loader):
        start_time = time.time()
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        net.train()
        # 前向传播
        y_pred = net(inputs)
        # 计算损失
        loss = criterion(y_pred, labels)
        losses += loss.item()                    # 收集每个batch的loss
        
        # 梯度清零
        optimizer.zero_grad()
        # 计算梯度
        loss.backward()
        # 更新参数
        optimizer.step()

    train_loss = losses/(i+1)                    # 用所有批次loss的均值作为这一个epoch的train_loss
    loss_curse.append(train_loss)

    # 验证集，这里暂时用testset代替validation
    # 用验证集的目的：利用模型没见过的验证集，验证模型在这一epoch的效果，根据根据这个效果调整超参数，如根据loss调整learning_rate，甚至epoch。
    # 正常情况下，用测试数据在训练时验证模型是不合理的。由于模型最终上线是去处理它没见过的测试数据，而用于验证的数据帮我们设置了超参数，因此，训练出来的模型是会拟合验证时候的数据。
    val_loss = compute_val_loss(net, test_loader, criterion, device)
    val_loss_curse.append(val_loss)

    if val_loss <= min_val_loss:                                   # 监控val_loss，如果连续5步不下降，则中断训练
        min_test_loss = val_loss
        torch.save(net, './' + model + str(epoch) + '.pt')
        count = 0
    else:
        count+=1
        if count >= 5:
            print('prevent overfitting...')
            break

    lr = optimizer.param_groups[0]['lr']
    lr_curse.append(lr)
    print('Lr:', lr)
    cos_lr.step()

    print('Epoch:', epoch, '---', 'train loss:', loss.item(), '---', 'val_loss:', val_loss, 'time:', time.time()-start_time)


# model saving
print('model saving...')
torch.save(net, './' + model + '_cifar.pt')
net = torch.load('./' + model + '_cifar.pt')

# testing
print('testing......')
# 测试
with torch.no_grad():
    net.eval()
    correct = 0
    total = 0

    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        out = net(images)
        _, predicted = torch.max(out.data, 1)           # 输出是10个数，分别代表每一类的概率。(max_value, index)=torch.max(input,dim)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    #输出识别准确率
    print('10000测试图像 准确率:{:.4f}%'.format(100 * correct / total))

# train_loss, val_loss
plt.figure()
plt.plot(range(len(loss_curse)), loss_curse, 'b', label='train')
plt.plot(range(len(val_loss_curse)), val_loss_curse, 'r', label='val')
plt.title(model)
plt.legend()
plt.show()

# 绘制学习率曲线
plt.figure()
plt.plot(range(len(lr_curse)), lr_curse, 'b', label='lr')
plt.legend()
plt.show()
