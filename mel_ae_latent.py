import torch
import torch.nn as nn
import torch.optim as optim
from AutoEncoder.ISIC_CAE import Encoder_latent_isic, Decoder_latent_isic, latent_AE
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([
        transforms.RandomRotation(180),
        transforms.RandomResizedCrop(
            size=224, scale=(0.4, 1.6), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
    ])

mel_folder = 'E:\\Universidad\\Master_UPC\\Segundo_anho\\TFM\\Divided_dataset\\test\\AE'
trainset = torchvision.datasets.ImageFolder(mel_folder, transform=transform)
batch_size = 1
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

enc = Encoder_latent_isic()
dec = Decoder_latent_isic()
model = latent_AE(enc, dec)

loss_fn = nn.BCELoss()
opt = optim.SGD(model.parameters(), lr=0.01)

loss_arr = []
loss_train = 0
for epoch in range(60):
    loss_train = 0
    for i, (data,_) in enumerate(trainloader):


        # training steps for normal model
        opt.zero_grad()
        outputs = model(data)
        # print(type(outputs))
        # print(type(data))
        loss = loss_fn(outputs, data)
        loss.backward()
        opt.step()
        loss_train += loss.item()

        
        # if i%100 == 0:
        #     print("Iter: {} batch loss: {}".format(i, loss.item()))
    loss_arr.append(loss_train/len(trainloader))    
    print('Epoch: {} Loss: {}'.format(epoch, loss_train/len(trainloader)))
    if epoch%15 == 0:
        chk = {
            'model' : model.state_dict(),
            'optim' : opt.state_dict(),
            'epoch' : epoch,
            'train_loss' : loss_arr
        }
        torch.save(chk, 'E:\\Universidad\\Master_UPC\\Segundo_anho\\TFM\\Code\\AutoEncoder\\autoencoder_latent.pth')
