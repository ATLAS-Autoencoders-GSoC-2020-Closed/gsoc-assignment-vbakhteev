import time

import torch
from tqdm.notebook import tqdm


def do_epoch(model, device, criterion, loader, epoch, phase='train', optimizer=None):
    running_loss = 0
    preds, labels = [], []
    is_train = phase == 'train'
    bs = loader.batch_size

    start = time.strftime("%H:%M:%S")
    msg = 'Loss: {:.4f}'
    msg_start = f'Starting epoch: {epoch} | phase: {phase} | ⏰: {start}'
    print(msg_start)

    model.train(is_train)
    loader_iter = tqdm(loader, leave=False)
    for itr, [x] in enumerate(loader_iter):
        x = x.to(device, non_blocking=True).float()
        decoded_x, encoded_x = model(x)
        loss = criterion(decoded_x, x)
        
        if is_train:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item() * bs
        preds.append(decoded_x.cpu())
        labels.append(x.cpu())

        loader_iter.set_description(
            msg.format(running_loss/bs/(itr+1))
        )

    preds = torch.cat(preds)
    labels = torch.cat(labels)
    running_loss /= len(loader) * bs
    msg = phase + ' ' + msg.format(running_loss)
    print(msg)

    return running_loss, preds, labels


def predict(model, device, loader):
    preds, labels = [], []
    model.eval()

    with torch.no_grad():
        for itr, [x] in enumerate(loader):
            x = x.to(device, non_blocking=True).float()
            decoded_x, encoded_x = model(x)
            preds.append(decoded_x.cpu())
            labels.append(x.cpu())

    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()

    return preds, labels
