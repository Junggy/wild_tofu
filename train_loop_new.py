import torch,os,torchvision,sys
from torch.utils.data import DataLoader
from utilities import *
from torch.optim import lr_scheduler
from losses import *
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from networks.main2 import ToFNetFusion

from train_setup import _setup_dict

def main(setup_name):
    setup = _setup_dict[setup_name]
    mode = os.path.join("results",setup_name)

    print(mode)

    if not os.path.exists(mode):
        os.mkdir(mode)

    checkpoints_dir = mode + "/checkpoints"
    if not os.path.exists(checkpoints_dir):
        os.mkdir(checkpoints_dir)

    logs_dir = mode + "/logs"
    if not os.path.exists(logs_dir):
        os.mkdir(logs_dir)

    val_dir = mode + "/inferece"
    if not os.path.exists(val_dir):
        os.mkdir(val_dir)

    test_dir = mode + "/test"
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    pretrain_dir = mode + "/pretrain"
    if not os.path.exists(pretrain_dir):
        os.mkdir(pretrain_dir)

    train_base_list = [setup["base"] + '/dataset/20201027-180941', setup["base"] + '/dataset/20201027-181503',
                       setup["base"] + '/dataset/20201027-181809']
    val_base_list = [setup["base"] + '/dataset/20201027-181125']

    crop_size = setup['crop_size']
    img_size = [480, 640]

    train_batch = setup['train_batch']
    val_batch = setup['val_batch']

    dmin = setup['dmin']
    dmax = setup['dmax']

    train_data = ToFDataset(train_base_list, "train", img_size, crop_size=crop_size, flip=False, filtered=setup["filtered"])
    train_loader = DataLoader(train_data, batch_size=train_batch, shuffle=True, drop_last=True)

    val_data = ToFDataset(val_base_list, "val", img_size, None, flip=False, filtered=setup['filtered'])
    val_loader = DataLoader(val_data, batch_size=val_batch, shuffle=False, drop_last=True)

    network = ToFNetFusion(dmin=dmin, dmax=dmax, setup=setup, fusion=setup['fusion'])
    network = network.cuda()

    optimizer = torch.optim.Adam(network.parameters(), setup['lr'])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=setup['step_size'], gamma=setup['gamma'])

    if crop_size == None: crop_size = img_size

    upscale_images_val = UpscaleImages(img_size)
    upscale_images_train = UpscaleImages(crop_size)

    multiscale_l1_loss = MultiscaleL1Loss(dmin, dmax)

    warp_stereo_train = InverseWarp(img_size + [3], img_size + [3], K_tof.cuda(), K_tof.cuda(), train_batch)
    warp_stereo_val = InverseWarp(img_size + [3], img_size + [3], K_tof.cuda(), K_tof.cuda(), val_batch)

    warp_tof_train = InverseWarp(img_size + [3], img_size + [3], K_tof.cuda(), K_tof.cuda(), train_batch)
    warp_tof_val = InverseWarp(img_size + [3], img_size + [3], K_tof.cuda(), K_tof.cuda(), val_batch)

    if setup['checkpoint'] == None:
        step = 0
        val_step = 0
    else:
        checkpoint = torch.load(os.path.join(checkpoints_dir, setup['checkpoint']))
        network.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        step = checkpoint["step"]
        val_step = checkpoint["val_step"]

    writer = SummaryWriter(logs_dir)

    network.train()

    for _ in range(10000):

        for dataset_out_ in train_loader:

            if step % setup['val_every'] == 0:

                print("start_val, step :", val_step)
                network.eval()

                if setup['rgb_l1']['weight'] != 0: rgb_l1_val_sum = 0
                if setup['tof_l1']['weight'] != 0: tof_l1_val_sum = 0
                if setup['rgb_smooth']['weight'] != 0: rgb_smooth_val_sum = 0
                if setup['rgb_self']['weight'] != 0: rgb_self_val_sum = 0
                if setup['tof_smooth']['weight'] != 0: tof_smooth_val_sum = 0
                if setup['tof_self']['weight'] != 0: tof_self_val_sum = 0

                for val_out_ in val_loader:
                    optimizer.zero_grad()
                    val_out = [each_elem.cuda() for each_elem in val_out_[:-1]]

                    depth, depth_warped, left, right, tof = val_out
                    network_in = torch.cat([left, tof], 1)

                    with torch.set_grad_enabled(False):
                        rgb_depths, tof_depths = network(network_in)

                        rgb_depths_upscaled = upscale_images_val(rgb_depths)
                        tof_depths_upscaled = upscale_images_val(tof_depths)

                        if setup['rgb_l1']['weight'] != 0:
                            rgb_l1_loss_pre, rgb_l1_loss = multiscale_l1_loss(depth, rgb_depths_upscaled)
                            rgb_l1_val_sum += rgb_l1_loss

                        if setup['tof_l1']['weight'] != 0:
                            tof_l1_loss_pre, tof_l1_loss = multiscale_l1_loss(depth_warped, tof_depths_upscaled)
                            tof_l1_val_sum += tof_l1_loss

                        if setup['rgb_smooth']['weight'] != 0:
                            rgb_smooth_loss_pre, rgb_smooth_loss = edge_aware_loss(left, rgb_depths[-1])
                            rgb_smooth_val_sum += rgb_smooth_loss

                        if setup['rgb_self']['weight'] != 0:
                            warped_left, rgb_self_loss_pre, rgb_self_loss = \
                                warp_stereo_val.stereo_self_supervised_loss(rgb_depths_upscaled, left, right,
                                                                            left_to_right.cuda())
                            rgb_self_val_sum += rgb_self_loss

                        if setup['tof_self']['weight'] != 0:
                            warped_from_left, warped_from_right, tof_self_loss_pre, tof_self_loss, mask = \
                                warp_tof_val.tof_rgb_self_supervised_loss(tof_depths_upscaled, left, right,
                                                                          ToF_to_RS.cuda(), ToF_to_right.cuda())
                            tof_self_val_sum += tof_self_loss

                            if setup['tof_smooth']['weight'] != 0:
                                pass

                    save_validation_images(rgb_depths[-1], tof_depths[-1], val_out_[-1], val_batch, val_dir, val_step)

                if setup['rgb_l1']['weight'] != 0:
                    rgb_l1_val_sum /= len(val_loader)
                    writer.add_scalar("val/rgb_depth", rgb_l1_val_sum.item(), val_step)
                    print("rgb l1 :", rgb_l1_val_sum.item())

                if setup['rgb_smooth']['weight'] != 0:
                    rgb_smooth_val_sum /= len(val_loader)
                    writer.add_scalar("val/rgb_smooth", rgb_smooth_val_sum.item(), val_step)
                    print("rgb smoothness :", rgb_smooth_val_sum.item())

                if setup['rgb_self']['weight'] != 0:
                    rgb_self_val_sum /= len(val_loader)
                    writer.add_scalar("val/rgb_self", rgb_self_val_sum.item(), val_step)
                    print("rgb self :", rgb_self_val_sum.item())

                if setup['tof_l1']['weight'] != 0:
                    tof_l1_val_sum /= len(val_loader)
                    writer.add_scalar("val/tof_depth", tof_l1_val_sum.item(), val_step)
                    print("tof l1 :", tof_l1_val_sum.item())

                if setup['tof_self']['weight'] != 0:
                    tof_self_val_sum /= len(val_loader)
                    writer.add_scalar("val/tof_self", tof_self_val_sum.item(), val_step)
                    print("tof self :", tof_self_val_sum.item())

                    if setup['tof_smooth']['weight'] != 0:
                        pass

                val_step += 1

                print("start train")
                network.train()

            optimizer.zero_grad()
            dataset_out = [each_elem.cuda() for each_elem in dataset_out_[:-1]]
            depth, depth_warped, left, right, tof = dataset_out
            network_in = torch.cat([left, tof], 1)

            with torch.set_grad_enabled(True):

                rgb_depths, tof_depths = network(network_in)

                total_loss = 0

                rgb_depths_upscaled = upscale_images_train(rgb_depths)
                tof_depths_upscaled = upscale_images_train(tof_depths)

                if setup['rgb_l1']['weight'] != 0:
                    rgb_l1_loss_pre, rgb_l1_loss = multiscale_l1_loss(depth, rgb_depths_upscaled)
                    writer.add_scalar("train/rgb_depth", rgb_l1_loss.item(), step)
                    total_loss += setup['rgb_l1']['weight'] * rgb_l1_loss

                if setup['tof_l1']['weight'] != 0:
                    tof_l1_loss_pre, tof_l1_loss = multiscale_l1_loss(depth_warped, tof_depths_upscaled)
                    writer.add_scalar("train/tof_depth", tof_l1_loss.item(), step)
                    total_loss += setup['tof_l1']['weight'] * tof_l1_loss

                if setup['rgb_self']['weight'] != 0:
                    warped_left, rgb_self_loss_pre, rgb_self_loss = \
                        warp_stereo_train.stereo_self_supervised_loss(rgb_depths_upscaled, left, right,
                                                                      left_to_right.cuda())
                    writer.add_scalar("train/rgb_self", rgb_self_loss.item(), step)
                    total_loss += setup['rgb_self']['weight'] * rgb_self_loss

                if setup['rgb_smooth']['weight'] != 0:
                    rgb_smooth_loss_pre, rgb_smooth_loss = edge_aware_loss(left, rgb_depths_upscaled[-1])
                    writer.add_scalar("train/rgb_smooth", rgb_smooth_loss.item(), step)
                    total_loss += setup['rgb_smooth']['weight'] * rgb_smooth_loss

                if setup['tof_self']['weight'] != 0:
                    warped_from_left, warped_from_right, tof_self_loss_pre, tof_self_loss, mask = \
                        warp_tof_train.tof_rgb_self_supervised_loss(tof_depths_upscaled, left, right, ToF_to_RS.cuda(),
                                                                    ToF_to_right.cuda())
                    writer.add_scalar("train/tof_self", tof_self_loss.item(), step)
                    total_loss += setup['tof_self']['weight'] * tof_self_loss

                    if setup['tof_smooth']['weight'] != 0:
                        pass

                total_loss.backward()

                optimizer.step()
                scheduler.step()

            if step % setup["save_every"] == 0 or step == setup["stop_at"]:
                checkpoint_path = os.path.join(checkpoints_dir, "{0:08d}_th_iter.pth".format(val_step))
                torch.save({'step': step, 'val_step': val_step, "model_state_dict": network.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict()}, checkpoint_path)

            # print(step, rgb_self_loss.item(), rgb_smooth_loss.item())

            step += 1
            if step >= setup['stop_at']: break

        if step > setup['stop_at']: break

if __name__ == "__main__":

    if not os.path.exists("./results"):
        os.mkdir("./results")

    args = sys.argv
    main(args[1])
