import argparse
import numpy as np
import random
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=1200, help="number of epochs of training")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.1, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=32, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=200, help="interval between image samples")
parser.add_argument('--nz', type=int, default=64, help="size of the latent z vector used as the generator input.")
opt = parser.parse_args()
# print(opt)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

batch_size = 5
dropout_level = 0.05
nz = opt.nz
img_shape = (9, 1500)
T = 3.0

def weights_init(m):
    if isinstance(m, nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)   

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.nz = opt.nz
        self.layer1 = nn.Sequential(
            nn.Linear(self.nz, 640),
            nn.PReLU()
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=16, out_channels=16, kernel_size=22, stride=4), 
            nn.PReLU()
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=16, out_channels=16, kernel_size=18, stride=2),
            nn.PReLU()
        )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=16, out_channels=2, kernel_size=16, stride=4),
            nn.Sigmoid()
        )
 
    def forward(self, z):
        out = self.layer1(z)
        out = out.view(out.size(0), 16, 40)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=16, kernel_size=10, stride = 2),
            nn.BatchNorm1d(num_features=16),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(2))
        self.dense_layers = nn.Sequential(
            nn.Linear(5968, 600),
            nn.LeakyReLU(0.2),
            nn.Linear(600, 1))

    def forward(self, x):
        out = self.layer1(x)
        out = out.view(out.size(0), -1)
        out = self.dense_layers(out)     
        return out

# Loss weight for gradient penalty
lambda_gp = 10

# Initialize generator and discriminator
discriminator = Discriminator()
generator = Generator()
discriminator.apply(weights_init)
generator.apply(weights_init)

discriminator.to(device)
generator.to(device)
    
def wgan(datatrain, label, nseed):

    random.seed(nseed)
    np.random.seed(nseed)
    torch.manual_seed(nseed)
    torch.cuda.manual_seed(nseed)

    datatrain = torch.from_numpy(datatrain)
    label = torch.from_numpy(label)

    dataset = torch.utils.data.TensorDataset(datatrain, label)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    
    def compute_gradient_penalty(D, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = Tensor(np.random.random((real_samples.size(0), 1, 1)))
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        # print (interpolates.shape)
        d_interpolates = D(interpolates)
        fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    # ----------
    #  Training
    # ----------
    new_data = []
    batches_done = 0
    discriminator.train()
    generator.train()
    for epoch in range(opt.n_epochs):
        for i, data in enumerate(dataloader, 0):

            imgs, _ = data
            imgs = imgs.cuda()
            
            # Configure input
            real_data = imgs.type(Tensor)

            if i % 1 == 0: 
                # ---------------------
                #  Train Discriminator
                # ---------------------
                optimizer_D.zero_grad()
                # Sample noise as generator input
                z = torch.randn(imgs.shape[0], nz).cuda()  
                # Generate a batch of images
                fake_imgs = generator(z)
                # Real images
                real_validity = discriminator(real_data)
                # Fake images
                fake_validity = discriminator(fake_imgs)
                # Gradient penalty
                gradient_penalty = compute_gradient_penalty(discriminator, real_data.data, fake_imgs.data)
                # Adversarial loss
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
                d_loss.backward()
                optimizer_D.step()
            
            # Train the generator every n_critic steps
            if i % opt.n_critic == 0:
                optimizer_G.zero_grad()
                # -----------------
                #  Train Generator
                # -----------------
                z = torch.randn(imgs.shape[0], nz).cuda() 
            # Generate a batch of images
                fake_imgs = generator(z)
                # Train on fake images
                fake_validity = discriminator(fake_imgs)
                g_loss = -torch.mean(fake_validity)
                g_loss.backward()
                optimizer_G.step()

                # print(
                #     "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                #     % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
                # )

    discriminator.eval()
    generator.eval()
    for epoch in range(100):
        for i, data in enumerate(dataloader, 0):
            imgs, _ = data
            imgs = imgs.cuda()
            z = torch.randn(imgs.shape[0], nz).cuda()  
            # Generate a batch of images
            fake_imgs = generator(z)
            # Train on fake images
            fake_validity = discriminator(fake_imgs)
            g_loss = -torch.mean(fake_validity) 
            # print(
            #         "[Epoch %d/%d] [Batch %d/%d] [G loss: %f]"
            #         % (epoch, opt.n_epochs, i, len(dataloader), g_loss.item())
            # )
            if i % opt.sample_interval == 0:
                # print (batches_done % opt.sample_interval) 
                fake_data = fake_imgs.data[:25].cpu().numpy()
                new_data.append(fake_data)

    new_data = np.concatenate(new_data) 
    new_data = np.asarray(new_data)
    return new_data
    