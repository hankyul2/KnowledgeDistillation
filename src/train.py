from easydict import EasyDict as edict


def run(args):
    print('run train')


if __name__ == '__main__':
    args = edict({
        'gpu_id':'',
    })

    run(args)
