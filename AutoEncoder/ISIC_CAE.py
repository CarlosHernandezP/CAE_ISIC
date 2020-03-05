import torch.nn as nn


class MyAE(nn.Module):
    def __init__(self):
        super(MyAE, self).__init__()
        self.encod1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), #Research about the "groups" hyperparameter
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(32)
        )
        
        
        self.encod2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64)
        )

        self.encod3 = nn.Sequential(
            nn.Conv2d(64, 128, 5, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128)
        )

        self.decod1 = nn.Sequential(
            nn.ConvTranspose2d(128,64, 3, stride=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64)
        )

        self.decod2 = nn.Sequential(
            nn.ConvTranspose2d(64,32, 3, stride=2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(32)
        )

        self.decod3 = nn.Sequential(
            nn.ConvTranspose2d(32,1, 2, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self,x):
        # Encoder
        
        x = self.encod1(x)
        
        x = self.encod2(x)
        x = self.encod3(x)

        # Decoder
        x = self.decod1(x)
        x = self.decod2(x)
        x = self.decod3(x)


        return x

class Encoder_latent(nn.Module):
    def __init__(self):
        super().__init__()
        self.encod1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), #Research about the "groups" hyperparameter
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(32)
        )        
        self.encod2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64)
        )
        self.encod3 = nn.Sequential(
            nn.Conv2d(64, 128, 5, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128)
        )
        self.decod1 = nn.Sequential(
            nn.ConvTranspose2d(128,64, 3, stride=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64)
        )
        self.decod2 = nn.Sequential(
            nn.ConvTranspose2d(64,32, 3, stride=2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(32)
        )

        self.decod3 = nn.Sequential(
            nn.ConvTranspose2d(32,3, 2, stride=2, padding=1),
            nn.Sigmoid()
        )

        self.latent_enc = nn.Linear(128*3*3,100)

    def forward(self,x):
        # Encoder
        
        x = self.encod1(x)
        x = self.encod2(x)

        x = self.encod3(x)

        x = x.reshape(x.shape[0], -1)
        x = self.latent_enc(x)


        return x

class Decoder_latent(nn.Module):
    def __init__(self):
        super().__init__()
        self.decod1 = nn.Sequential(
            nn.ConvTranspose2d(128,64, 3, stride=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64)
        )

        self.decod2 = nn.Sequential(
            nn.ConvTranspose2d(64,32, 3, stride=2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(32)
        )

        self.decod3 = nn.Sequential(
            nn.ConvTranspose2d(32,1, 2, stride=2, padding=1),
            nn.Sigmoid()
        )
        self.latent_dec = nn.Linear(100, 128*3*3)

    def forward(self,x):        
        # Decoder
        x = self.latent_dec(x)
        x = x.view(-1, 128, 3, 3)

        x = self.decod1(x)
        x = self.decod2(x)
        x = self.decod3(x)
        return x

class latent_AE(nn.Module):
    def __init__(self, enc, dec):
        super().__init__()
        self.encoder = enc
        self.decoder = dec

    def forward(self,x):
        x = self.encoder(x)
        predicted = self.decoder(x)
        return predicted



# CAE for isic
class MyAE_isic(nn.Module):
    def __init__(self):
        super().__init__()
        self.encod1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), #Research about the "groups" hyperparameter
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(32)
        )
        
        self.encod2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64)
        )

        self.encod3 = nn.Sequential(
            nn.Conv2d(64, 128, 5, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128)
        )

        self.decod1 = nn.Sequential(
            nn.ConvTranspose2d(128,64, 5, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64)
        )

        self.decod2 = nn.Sequential(
            nn.ConvTranspose2d(64,32, 5, stride=2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(32)
        )

        self.decod3 = nn.Sequential(
            nn.ConvTranspose2d(32,3, 2, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self,x):
        # Encoder
        # torch.Size([1, 3, 448, 448])
        x = self.encod1(x)
        # torch.Size([1, 32, 224, 224])
        x = self.encod2(x)
        # torch.Size([1, 64, 112, 112])
        x = self.encod3(x)
        # torch.Size([1, 128, 55, 55])
        # Decoder
        x = self.decod1(x)
        # torch.Size([1, 64, 111, 111])
        x = self.decod2(x)
        # torch.Size([1, 32, 225, 225])
        x = self.decod3(x)
        # torch.Size([1, 3, 448, 448])
        return x

class Encoder_latent_isic(nn.Module):
    def __init__(self):
        super().__init__()
        self.encod1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), #Research about the "groups" hyperparameter
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(32)
        )        
        self.encod2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64)
        )
        self.encod3 = nn.Sequential(
            nn.Conv2d(64, 128, 5, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128)
        )
        self.latent_enc = nn.Linear(128*27*27,1000)

    def forward(self,x):
        # Encoder
        
        x = self.encod1(x)
        x = self.encod2(x)

        x = self.encod3(x)
        print(x.shape)
        x = x.reshape(x.shape[0], -1)
        x = self.latent_enc(x)


        return x

class Decoder_latent_isic(nn.Module):
    def __init__(self):
        super().__init__()
        self.decod1 = nn.Sequential(
            nn.ConvTranspose2d(128,64, 5, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64)
        )

        self.decod2 = nn.Sequential(
            nn.ConvTranspose2d(64,32, 5, stride=2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(32)
        )

        self.decod3 = nn.Sequential(
            nn.ConvTranspose2d(32,3, 2, stride=2, padding=1),
            nn.Sigmoid()
        )
        self.latent_dec = nn.Linear(1000, 128*27*27)

    def forward(self,x):        
        # Decoder
        x = self.latent_dec(x)
        x = x.view(-1, 128, 27, 27)
        print(x.shape)
        x = self.decod1(x)
        print(x.shape)
        x = self.decod2(x)
        print(x.shape)
        x = self.decod3(x)
        print(x.shape)
        return x