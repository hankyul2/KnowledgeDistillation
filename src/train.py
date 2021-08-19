
from easydict import EasyDict as edict

from src.dataset import get_dataset, convert_to_dataloader


def run(args):
    # step 1. load dataset
    train_ds, valid_ds, test_ds = get_dataset(args.dataset)
    train_dl, = convert_to_dataloader([train_ds], batch_size=args.batch_size, num_workers=args.num_workers, train=True)
    valid_dl, test_dl = convert_to_dataloader([valid_ds, test_ds], batch_size=args.batch_size, num_workers=args.num_workers, train=False)

    # step 2. load model









if __name__ == '__main__':
    # this is for jupyter users
    args = edict({
        'gpu_id':'',
    })
    run(args)
