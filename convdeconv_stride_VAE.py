
# coding: utf-8

# In[ ]:

from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image


# In[ ]:

seed = 1
batch_size = 128
epochs = 10
log_interval = 10


# In[ ]:

HAVE_CUDA=True


# In[ ]:

kwargs = {'num_workers': 1, 'pin_memory': True} if HAVE_CUDA else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)


# In[ ]:

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, stride=2) # 1 inp channel, 10 output channels
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, stride=2) # 
        self.conv2_drop = nn.Dropout2d() 

        
        self.fc1 = nn.Linear(320, 100)
        self.fc21 = nn.Linear(100, 20)
        self.fc22 = nn.Linear(100, 20)
        self.fc3 = nn.Linear(20, 100)
        self.fc4 = nn.Linear(100, 784)
        self.fc28 = nn.Linear(784,784)
        
        self.d_fc3 = nn.Linear(20, 100)
        self.d_fc4 = nn.Linear(100, 320)
        self.unpool = nn.MaxUnpool2d(2,stride=2)
        self.pool1   = nn.MaxPool2d(1,stride=1, return_indices=True)
        self.deconv1 = nn.ConvTranspose2d(20,10,kernel_size=5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(10,1, kernel_size=5, stride=2)
        self.Dfc25= nn.Linear(625, 784)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
          std = logvar.mul(0.5).exp_()
          eps = Variable(std.data.new(std.size()).normal_())
          return eps.mul(std).add_(mu)
        else:
          return mu

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))
    
    def decode_deconv(self,z):
        h3 = self.relu(self.d_fc3(z)) # 20->100
        h4 = self.relu(self.d_fc4(h3)) #100->320
        h4 = h4.view(-1, 20, 4, 4) #20 channels
        h11 = self.relu(self.deconv1(h4))
        h25 = self.relu(self.deconv2(h11))
        h28 = self.Dfc25(h25.view(-1,625))
        return self.sigmoid(h28.view(-1,784))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        mu, logvar = self.encode(x.view(-1, 320))
        z = self.reparameterize(mu, logvar)
        return self.decode_deconv(z), mu, logvar


# In[ ]:

model = VAE()
if HAVE_CUDA:
    model.cuda()


# In[ ]:

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784))

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Normalise by same number of elements as in reconstruction
    KLD /= batch_size * 784

    return BCE + KLD


# In[ ]:

optimizer = optim.Adam(model.parameters(), lr=1e-3)


# In[ ]:

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data)
        if HAVE_CUDA:
            data = data.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data[0] / len(data)))

    print('====> TRAIN Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))
    loss_val =  train_loss/len(train_loader.dataset)
    return loss_val


# In[ ]:

def test(epoch):
    model.eval()
    test_loss = 0
    for i, (data, _) in enumerate(test_loader):
        if HAVE_CUDA:
            data = data.cuda()
        data = Variable(data, volatile=True)
        recon_batch, mu, logvar = model(data)
        test_loss += loss_function(recon_batch, data, mu, logvar).data[0]
        if i == 0:
          n = min(data.size(0), 8)
          comparison = torch.cat([data[:n],
                                  recon_batch.view(batch_size, 1, 28, 28)[:n]])
          save_image(comparison.data.cpu(),
                     'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


# In[ ]:

losses = []
for epoch in range(1, 1 + 100):
    train_loss = train(epoch)
    print('train_loss', train_loss)
    losses.append(train_loss)
    test(epoch)
    sample = Variable(torch.randn(64, 20))
    if HAVE_CUDA:
       sample = sample.cuda()
    sample = model.decode_deconv(sample).cpu()
    save_image(sample.data.view(64, 1, 28, 28),
               'results/sample_' + str(epoch) + '.png')


# In[ ]:

import matplotlib.pyplot as plt


# In[ ]:

print(losses)


# In[ ]:

plt.plot(losses)


# In[ ]:

plt.show()


# In[ ]:



