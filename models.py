import torch
from torch import nn
from torchvision import models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_cnn=False) -> None:
        super().__init__()

        self.train_cnn = train_cnn
        self.inception = models.inception_v3(pretrained=True, aux_logits=False)
        # Replace the last layer
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=.5)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.inception(images)

        # Fine-tuning the fully connected layer
        for name, param in self.inception.named_parameters():
            if 'fc.weight' in name or 'fc.bias' in name:
                param.requires_grad = True
            else:
                param.requires_grad = self.train_cnn

        return self.dropout(self.relu(features))


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.liner = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(p=.5)

    def forward(self, features: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        embeddings = self.dropout(self.embed(captions))
        embeddings = torch.cat((features.unsqueeze(0), embeddings))
        hiddens, _ = self.lstm(embeddings)
        return self.liner(hiddens)


class CNN2RNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super().__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images)
        return self.decoder(features, captions)

    def caption_image(self, image, vocabulary, max_length=50):
        result_caption = []
        with torch.no_grad():
            x = self.encoder(image).unsqueeze(0)
            states = None

            for _ in range(max_length):
                hiddens, states = self.decoder.lstm(x, states)
                output = self.decoder.liner(hiddens.squeeze(0))
                predicted = output.argmax(1)
                result_caption.append(predicted.item())
                x = self.decoder.embed(predicted).unsqueeze(0)

                if vocabulary.itos[predicted.item()] == '<EOS>':
                    break

        return [vocabulary.itos[idx] for idx in result_caption]
