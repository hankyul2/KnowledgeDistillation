import torch
from easydict import EasyDict as edict

from src.dataset import get_dataset, convert_to_dataloader
from src.resnet import get_model


def run(args):
    # step 1. load dataset
    train_ds, valid_ds, test_ds = get_dataset(args.dataset)
    train_dl, = convert_to_dataloader([train_ds], batch_size=args.batch_size, num_workers=args.num_workers, train=True)
    valid_dl, test_dl = convert_to_dataloader([valid_ds, test_ds], batch_size=args.batch_size, num_workers=args.num_workers, train=False)

    # step 2. load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(args.model_name, device=device, nclass=1000, pretrained=args.pre_trained)








if __name__ == '__main__':
    # this is for jupyter users
    args = edict({
        'gpu_id':'',
    })
    run(args)
