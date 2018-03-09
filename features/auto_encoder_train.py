import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
from images.loader import SingleFolderDataSet
from torchvision import transforms
from torchvision import models
# torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 1
BATCH_SIZE = 100
LR = 0.01         # learning rate
INPUT_SHAPE = (224, 224)


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.Tanh(),
            nn.Linear(4096, 512),
        )

        self.decoder = nn.Sequential(
            nn.Linear(512, 4096),
            nn.Tanh(),
            nn.Linear(4096, 256 * 6 * 6),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


def train_auto_encoder(image_dir, input_size, save_path, if_gpu):
    train_data = SingleFolderDataSet(image_dir, transform=transforms.Compose([
        transforms.Resize(size=input_size),
        transforms.ToTensor(),
    ]))
    # Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

    auto_encoder = AutoEncoder()
    auto_encoder.load_state_dict(torch.load(save_path))
    alex = models.alexnet(pretrained=True)

    if if_gpu:
        alex.cuda()
        auto_encoder.cuda()

    optimizer = torch.optim.Adam(auto_encoder.parameters(), lr=LR)
    loss_func = nn.MSELoss()

    for epoch in range(EPOCH):
        for step, (x, y,) in enumerate(train_loader):

            var_data = Variable(x)
            if if_gpu:
                var_data = var_data.cuda()
            feature = alex.features(var_data)

            b_x = Variable(feature.data.view(-1, 256 * 6 * 6))
            b_y = Variable(feature.data.view(-1, 256 * 6 * 6))  # batch y, shape (batch, 256 * 6 * 6)

            if if_gpu:
                b_x = b_x.cuda()
                b_y = b_x.cuda()

            encoded, decoded = auto_encoder(b_x)

            loss = loss_func(decoded, b_y)  # mean square error
            optimizer.zero_grad()           # clear gradients for this training step
            loss.backward()                 # back propagation, compute gradients
            optimizer.step()                # apply gradients

            if step % 100 == 0:
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0])

    # torch.save(auto_encoder, save_path)
    # the_model = torch.load(save_path)
    torch.save(auto_encoder.state_dict(), save_path)
    auto_encoder.load_state_dict(torch.load(save_path))
    print(auto_encoder)


if __name__ == '__main__':

    save_path = './auto_encoder.torch'

    the_model = AutoEncoder()
    the_model.load_state_dict(torch.load(save_path))
    print(the_model)

    test_images = '/home/xinyan/programs/data/landmark/test'
    train_images = '/home/xinyan/programs/data/landmark/train'

    train_auto_encoder(train_images, INPUT_SHAPE, save_path,  False)
