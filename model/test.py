import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# pyplot으로 시각화 할때 뜬 에러 처리
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)
print("device : ", device)

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = torch.nn.Linear(7*7*64, 10, bias=True)
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

print("model load...")
model = torch.load('./mnist_model.pt')

print("ready to model test...")
batch_size = 100

test_set = torchvision.datasets.MNIST(
    root = './data/MNIST',
    train = False,
    download = True,
    transform = transforms.Compose([
        transforms.ToTensor() # 데이터를 0에서 255까지 있는 값을 0에서 1사이 값으로 변환
    ])
)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

for data, target in test_loader:
    data = data.to(device)
    target = target.to(device)

print("start to test...")
i = 5
for i in range(i, i+5):
    test_data = torch.unsqueeze(data[i], dim=0)
    test = model(test_data)
    print(torch.argmax(test))
    print(target[i])

    plt.imshow(data[i].view(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()
