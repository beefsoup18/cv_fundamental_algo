import torch
import torch.nn as nn
import torch.optim as optim
# import torchvision.transforms as transforms
from torch.utils.data import DataLoader
# from torchvision.datasets import CIFAR10
# from torchsummary import summary
# from einops.layers.torch import Rearrange
from torchvision import transforms, datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 1000

# 定义模型
class ViT(nn.Module):

    def __init__(self, image_size, patch_size, emb_size, num_heads, num_layers, num_classes):
        super(ViT, self).__init__()
        assert image_size % patch_size == 0, "image size must be divisible by patch size"
        num_patches = (image_size // patch_size) ** 2
        self.patch_size = patch_size
        self.project_patch = nn.Conv2d(3, emb_size, kernel_size=patch_size, stride=patch_size, bias=False)   # obj
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        # self.pos_embeddings = nn.Parameter(torch.randn(num_patches + 1, 1, emb_size))  # [num_patches + 1, 1, embedding_size]
        self.pos_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, emb_size))  # [batch_size, num_patches + 1, 1, embedding_size]
        self.dropout = nn.Dropout(p=0.5)
        self.transformer_blocks = nn.Sequential(*[
            TransformerBlock(emb_size, num_heads, dropout_prob=0.5) for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(emb_size)
        self.fc = nn.Linear(emb_size, num_classes)

    def forward(self, x):
        x = self.project_patch(x)   # [batch_size, embedding_size, H, W] = [64, 768, 14, 14]
        b, embeds, h, w = x.shape
        x = x.view(b, embeds, -1).transpose(1, 2)   # [batch_size, h*w, embedding_size] = [64, 196, 768]
        pos_embeddings = self.pos_embeddings.detach().clone()
        # self.pos_embeddings = self.pos_embeddings.repeat(x.size(0), 1, 1, 1)  
        pos_embeddings = pos_embeddings.repeat(x.size(0), 1, 1, 1)
        self.pos_embeddings = nn.Parameter(pos_embeddings, requires_grad=True)   # [64, 197, 1, 768]

        cls_tokens = self.cls_token.expand(b, -1, -1)   # [64, 1, 768]
        x = torch.cat([cls_tokens, x], dim=1)  # [64, 197, 768]
        x = x + self.pos_embeddings
        x = self.dropout(x)

        # Add the extra dimension for layer norm
        x = x.unsqueeze(1)
        x = self.transformer_blocks(x)
        x = x.squeeze(1)
        x = self.layer_norm(x[:, 0])
        x = self.fc(x)
        return x


class TransformerBlock(nn.Module):

    def __init__(self, emb_size, num_heads, dropout_prob):
        super(TransformerBlock, self).__init__()
        self.multi_head_self_attention = nn.MultiheadAttention(emb_size, num_heads)
        self.layer_norm_1 = nn.LayerNorm(emb_size)
        self.feedforward = nn.Sequential(
            nn.Linear(emb_size, 4 * emb_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(4 * emb_size, emb_size),
            nn.Dropout(dropout_prob)
        )
        self.layer_norm_2 = nn.LayerNorm(emb_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        residual = x
        x = self.multi_head_self_attention(x, x, x)[0] + residual
        x = self.layer_norm_1(x)
        residual = x
        x = self.feedforward(x) + residual
        x = self.layer_norm_2(x)
        x = self.dropout(x)
        return x


# # 训练模型
# def train(model, optimizer, criterion, dataloader):
#     model.train()
#     for i, (images, labels) in enumerate(dataloader):
#         images, labels = images.to(device), labels.to(device)
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()


# # 测试模型
# def test(model, dataloader):
#     model.eval()
#     with torch.no_grad():
#         total = 0
#         correct = 0
#         for images, labels in dataloader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted ==


def train(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for batch_idx, (images, labels) in enumerate(train_loader):
        print(batch_idx, labels)
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        if batch_idx % 100 == 99:
            print('Batch {} Loss: {:.4f}'.format(batch_idx+1, running_loss/((batch_idx+1)*train_loader.batch_size)))
    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss


def test(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    epoch_loss = running_loss / len(test_loader.dataset)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViT(image_size=224, patch_size=16, emb_size=768, num_heads=12, num_layers=12, num_classes=10).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()
# train_loader = ...
# test_loader = ...

# define data augmentation and preprocessing function
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# load CIFAR10 dataset
# train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transforms)
# test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transforms)

# train_dataset = datasets.ImageFolder(root='~/data/1920i/raw/Tom_Jerry_Food_Adventures/All_1920i/', transform=train_transforms)
# test_dataset = datasets.ImageFolder(root='~/data/1920i/raw/Tom_Jerry_Food_Adventures/I_1920i/', transform=train_transforms)

train_dataset = datasets.ImageFolder(root='~/Downloads/cifar-10-python/dataset', transform=train_transforms)
test_dataset = datasets.ImageFolder(root='~/Downloads/cifar-10-python/dataset', transform=test_transforms)

# create DataLoader for mini-batch training and testing
batch_size = 64
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


for epoch in range(num_epochs):
    train_loss = train(model, train_loader, optimizer, criterion, device)
    test_loss, test_acc = test(model, test_loader, criterion, device)
    print(f"Epoch {epoch+1}/{num_epochs}: Train Loss={train_loss:.4f}, Test Loss={test_loss:.4f}, Test Acc={test_acc:.2f}%")