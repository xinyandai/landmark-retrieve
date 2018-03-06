import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
from images.loader import SingleFolderDataSet
from torchvision import transforms
from torchvision import models
# torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 10
BATCH_SIZE = 1
LR = 0.005         # learning rate
N_TEST_IMG = 5

INPUT_SHAPE = (224, 224)


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        # self.encoder = nn.Sequential(
        #     nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
        #     nn.ReLU(True),
        #     nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
        #     nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
        #     nn.ReLU(True),
        #     nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        # )
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
        #     nn.Tanh()
        # )

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Linear(256 * 6 * 6, 4096),
            nn.Tanh(),
            nn.Linear(4096, 512),
        )

        self.decoder = nn.Sequential(
            nn.Linear(512, 4096),
            nn.Tanh(),
            nn.Linear(4096, 256 * 6 * 6),
            # nn.Sigmoid(),  # compress to a range (0, 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


def train_auto_encoder(image_dir, input_size, save_path, if_gpu=False):
    train_data = SingleFolderDataSet(image_dir, transform=transforms.Compose([
        transforms.Resize(size=input_size),
        transforms.ToTensor(),
    ]))
    # Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

    auto_encoder = AutoEncoder()
    if if_gpu:
        auto_encoder.cuda()

    optimizer = torch.optim.Adam(auto_encoder.parameters(), lr=LR)
    loss_func = nn.MSELoss()

    for epoch in range(EPOCH):
        for step, (x, y,) in enumerate(train_loader):

            b_x = Variable(x)
            b_y = Variable(x)
            if if_gpu:
                b_x = b_x.cuda()
                b_y = b_y.cuda()

            encoded, decoded = auto_encoder(b_x)

            loss = loss_func(decoded, b_y)  # mean square error
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            if step % 10 == 0:
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0])

    torch.save(models.state_dict(), save_path)


# torch.save(models.state_dict(), save_path)
# the_model = TheModelClass(*args, **kwargs)
# the_model.load_state_dict(torch.load(PATH))


if __name__ == '__main__':

    test_images = '/home/xinyan/programs/data/landmark/test'
    train_images = '/home/xinyan/programs/data/landmark/train'

    train_auto_encoder(train_images, INPUT_SHAPE, './cnn_encoder.torch',  False)
