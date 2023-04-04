import torch.optim as optim
from torchtext.datasets import IMDB
from torchtext.data import Field, LabelField, BucketIterator

from transformer_demo import Transformer


TEXT = Field(tokenize='spacy', lower=True, include_lengths=True)
LABEL = LabelField(dtype=torch.float)

train_data, test_data = IMDB.splits(TEXT, LABEL)
TEXT.build_vocab(train_data, max_size=5000)
LABEL.build_vocab(train_data)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader, test_loader = BucketIterator.splits(
    (train_data, test_data), batch_size=32, device=device, sort_within_batch=True, repeat=False
)

model = TransformerClassifier(len(TEXT.vocab), 512, 128, 8, 256, 4, TEXT.vocab.stoi[TEXT.pad_token], len(LABEL.vocab))
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()

for epoch in range(10):
    model.train()
    for batch in train_loader:
        input_data, lengths = batch.text
        labels = batch.label
        
        optimizer.zero_grad()
        logits = model(input_data)
        loss = criterion(logits.squeeze(), labels)
        loss.backward()
        
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch in test_loader:
            input_data, lengths = batch.text
            labels = batch.label
            
            logits = model(input_data)
            loss = criterion(logits.squeeze(), labels)
            
            preds = (logits > 0).long()
            correct = (preds == labels.long()).sum().item()
            
            total_loss += loss.item() * len(batch)
            total_correct += correct
            total_samples += len(batch)
        
        acc = total_correct / total_samples
        avg_loss = total_loss / total_samples
        print(f"Epoch {epoch}, Test Loss: {avg_loss:.3f}, Test Accuracy: {acc:.3f}")
