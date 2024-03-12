import argparse
import torch
import os
import torch.nn.functional as F
from dataset import SKIN, HERDataset
from model import STco
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import AvgMeter, get_lr

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128, help='')
parser.add_argument('--max_epochs', type=int, default=100, help='')
parser.add_argument('--temperature', type=float, default=1., help='temperature')
parser.add_argument('--fold', type=int, default=0, help='fold')
parser.add_argument('--dim', type=int, default=785, help='spot_embedding dimension (# HVGs)')  # 171, 785
parser.add_argument('--image_embedding_dim', type=int, default=1024, help='image_embedding dimension')
parser.add_argument('--projection_dim', type=int, default=256, help='projection_dim ')
parser.add_argument('--dataset', type=str, default='her2st', help='dataset')


def load_data(args):
    if args.dataset == 'her2st':
        print(f'load dataset: {args.dataset}')
        train_dataset = HERDataset(train=True, fold=args.fold)
        train_dataLoader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_dataset = HERDataset(train=False, fold=args.fold)
        return train_dataLoader, test_dataset
    elif args.dataset == 'cscc':
        print(f'load dataset: {args.dataset}')
        train_dataset = SKIN(train=True, fold=args.fold)
        train_dataLoader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_dataset = SKIN(train=False, fold=args.fold)
        return train_dataLoader, test_dataset


def train(model, train_dataLoader, optimizer, epoch):
    loss_meter = AvgMeter()
    tqdm_train = tqdm(train_dataLoader, total=len(train_dataLoader))
    for batch in tqdm_train:
        batch = {k: v.cuda() for k, v in batch.items() if
                 k == "image" or k == "expression" or k == "position"}
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)
        tqdm_train.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer), epoch=epoch)


def save_model(args, model, test_dataset=None, examples=[]):
    os.makedirs(f"./model_result/{args.dataset}/{test_dataset.id2name[0]}", exist_ok=True)
    torch.save(model.state_dict(),
               f"./model_result/{args.dataset}/{test_dataset.id2name[0]}/best_{args.fold}.pt")

def main():
    args = parser.parse_args()
    for i in range(0, 32):
        args.fold = i
        print("当前fold:", args.fold)
        train_dataLoader, test_dataset = load_data(args)
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = STco(spot_embedding=args.dim, temperature=args.temperature,
                                 image_embedding=args.image_embedding_dim, projection_dim=args.projection_dim).cuda()

        optimizer = torch.optim.Adam(
            model.parameters(), lr=1e-4, weight_decay=1e-3
        )
        for epoch in range(args.max_epochs):
            model.train()
            train(model, train_dataLoader, optimizer, epoch)


        save_model(args, model, test_dataset=test_dataset)
        print("Saved Model")

main()
