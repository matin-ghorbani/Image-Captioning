import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import models
import utils as ul
import config
import loader


def train() -> None:
    transform = transforms.Compose([
        transforms.Resize(config.IMG_SHAPE),
        transforms.RandomCrop((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])

    train_loader, dataset = loader.get_loader(
        root_folder=config.ROOT_DIR,
        annotation_file=config.ANNOTATION_FILE,
        transform=transform,
        num_workers=2
    )

    torch.backends.cudnn.benchmark = True
    vocab_size = len(dataset.vocab)

    # TensorBoard
    writer = SummaryWriter('runs/flickr')
    step = 0

    model = models.CNN2RNN(
        embed_size=config.EMBED_SIZE,
        hidden_size=config.HIDDEN_SIZE,
        vocab_size=vocab_size,
        num_layers=config.NUM_LAYERS
    ).to(config.DEVICE)
    loss_fn = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi['<PAD>'])
    optimizer = optim.Adam(model.parameters(), lr=config.LR)

    if config.LOAD_MODEL:
        step = ul.load_checkpoint(torch.load(config.CHECKPOINT), model, optimizer)

    model.train()
    for epoch in range(1, config.NUM_EPOCHS + 1):
        ul.print_examples(model, config.DEVICE, dataset)

        if config.SAVE_MODEL:
            ul.save_checkpoint({
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': step
            })

        print(f'{epoch = }')

        imgs: torch.Tensor
        captions: torch.Tensor
        for imgs, captions in train_loader:
            imgs = imgs.to(config.DEVICE)
            captions = captions.to(config.DEVICE)

            # captions[:-1]: We want model to predicts the end token
            outputs: torch.Tensor = model(imgs, captions[:-1])
            loss = loss_fn(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))

            writer.add_scalar('Training Loss', loss.item(), global_step=step)
            step += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def main() -> None:
    train()


if __name__ == '__main__':
    main()
