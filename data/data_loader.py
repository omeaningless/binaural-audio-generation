import torch
from data.stereomulti_dataset import StereoDataset
def CreateDataLoader(opt):
    dataset = StereoDataset(opt)
    print("dataset [%s] was created" % (dataset.name()))
    print("#%s clips = %d" %(opt.mode, len(dataset)))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batchSize,
        shuffle=opt.mode=='train',
        num_workers=int(opt.nThreads),
        drop_last=opt.mode=='train'
    )

    return dataloader
