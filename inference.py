import torch,os,torchvision,sys
from torch.utils.data import DataLoader
from utilities import *
from torch.optim import lr_scheduler
from losses import *
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from networks.main2 import ToFNetFusion

from test_setup import _setup_dict

def main(setup_name):

    setup = _setup_dict[setup_name]
    val_dir = "testset/output"
    print(val_dir)

    if not os.path.exists(val_dir):
        os.mkdir(val_dir)

    val_base_list = [setup["base"] + '/dataset/20201027-181125']

    crop_size = None
    img_size = [480, 640]

    val_batch = setup['val_batch']

    dmin = setup['dmin']
    dmax = setup['dmax']

    val_data = ToFDataset(val_base_list, "val", img_size, None, flip=False, filtered=setup['filtered'],test=True)
    val_loader = DataLoader(val_data, batch_size=val_batch, shuffle=False, drop_last=True)

    network = ToFNetFusion(dmin=dmin, dmax=dmax, setup=setup, fusion=setup['fusion'])
    network = network.cuda()


    if crop_size == None: crop_size = img_size

    checkpoint = torch.load(setup['checkpoint'])
    network.load_state_dict(checkpoint["model_state_dict"])

    network.eval()

    for val_out_ in val_loader:
        val_out = [each_elem.cuda() for each_elem in val_out_[:-1]]

        left, tof = val_out
        network_in = torch.cat([left, tof], 1)

        with torch.set_grad_enabled(False):
            rgb_depths, tof_depths = network(network_in)

        save_validation_images(rgb_depths[-1], tof_depths[-1], val_out_[-1], val_batch, val_dir, None, is_test = True)

if __name__ == "__main__":

    args = sys.argv
    main(args[1])
