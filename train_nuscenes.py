import os
import time
import argparse
import numpy as np
import saverloader
from fire import Fire
from nets.segnet import Segnet
import utils.misc
import utils.improc
import utils.vox
import random
import nuscenesdataset 
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from torchvision.utils import save_image

random.seed(125)
np.random.seed(125)


# the scene centroid is defined wrt a reference camera,
# which is usually random
scene_centroid_x = 0.0
scene_centroid_y = 1.0 # down 1 meter
scene_centroid_z = 0.0

scene_centroid_py = np.array([scene_centroid_x,
                              scene_centroid_y,
                              scene_centroid_z]).reshape([1, 3])
scene_centroid = torch.from_numpy(scene_centroid_py).float()

XMIN, XMAX = -50, 50
ZMIN, ZMAX = -50, 50
YMIN, YMAX = -5, 5
bounds = (XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX)

Z, Y, X = 200, 8, 200

def requires_grad(parameters, flag=True):
    for p in parameters:
        p.requires_grad = flag
        
def fetch_optimizer(lr, wdecay, epsilon, num_steps, params):
    """ Create the optimizer and learning rate scheduler """
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wdecay, eps=epsilon)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, num_steps+100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler

class SimpleLoss(torch.nn.Module):
    def __init__(self, pos_weight):
        super(SimpleLoss, self).__init__()
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]), reduction='none')

    def forward(self, ypred, ytgt, valid):
        loss = self.loss_fn(ypred, ytgt)
        loss = utils.basic.reduce_masked_mean(loss, valid)
        return loss

def balanced_mse_loss(pred, gt, valid=None):
    pos_mask = gt.gt(0.5).float()
    neg_mask = gt.lt(0.5).float()
    if valid is None:
        valid = torch.ones_like(pos_mask)
    mse_loss = F.mse_loss(pred, gt, reduction='none')
    pos_loss = utils.basic.reduce_masked_mean(mse_loss, pos_mask*valid)
    neg_loss = utils.basic.reduce_masked_mean(mse_loss, neg_mask*valid)
    loss = (pos_loss + neg_loss)*0.5
    return loss
    
def run_model(model, loss_fn, d, device='cuda:0', sw=None):
    metrics = {}
    total_loss = torch.tensor(0.0, requires_grad=True).to(device)

    (
        imgs, rots, trans, intrins, pts0, extra0, pts, extra, 
        lrtlist_velo, vislist, tidlist, scorelist, 
        seg_bev_g, valid_bev_g, center_bev_g, offset_bev_g, radar_data, egopose,
        pix_masks, foreground_areas, incomplete_objects_areas)= d

    B0,T,S,C,H,W = imgs.shape
    assert(T==1)

    # eliminate the time dimension
    imgs = imgs[:,0]
    rots = rots[:,0]
    trans = trans[:,0]
    intrins = intrins[:,0]
    pts0 = pts0[:,0]
    extra0 = extra0[:,0]
    pts = pts[:,0]
    extra = extra[:,0]
    lrtlist_velo = lrtlist_velo[:,0]
    vislist = vislist[:,0]
    tidlist = tidlist[:,0]
    scorelist = scorelist[:,0]
    seg_bev_g = seg_bev_g[:,0]
    valid_bev_g = valid_bev_g[:,0]
    center_bev_g = center_bev_g[:,0]
    offset_bev_g = offset_bev_g[:,0]
    radar_data = radar_data[:,0]
    egopose = egopose[:,0]
    
    origin_T_velo0t = egopose.to(device) # B,T,4,4
    lrtlist_velo = lrtlist_velo.to(device)
    scorelist = scorelist.to(device)

    rgb_camXs = imgs.float().to(device)
    rgb_camXs = rgb_camXs - 0.5 # go to -0.5, 0.5

    seg_bev_g = seg_bev_g.to(device)
    valid_bev_g = valid_bev_g.to(device)
    center_bev_g = center_bev_g.to(device)
    offset_bev_g = offset_bev_g.to(device)

    pix_masks = pix_masks[:, 0]
    pix_masks = pix_masks.to(device)
    if model.module.supervise_height:
        foreground_areas = foreground_areas[:, 0]
        foreground_areas = foreground_areas.to(device)
        incomplete_objects_areas = incomplete_objects_areas[:, 0]
        incomplete_objects_areas = incomplete_objects_areas.to(device)

    xyz_velo0 = pts.to(device).permute(0, 2, 1)
    rad_data = radar_data.to(device).permute(0, 2, 1) # B, R, 19
    xyz_rad = rad_data[:,:,:3]
    meta_rad = rad_data[:,:,3:]

    B, S, C, H, W = rgb_camXs.shape
    B, V, D = xyz_velo0.shape

    __p = lambda x: utils.basic.pack_seqdim(x, B)
    __u = lambda x: utils.basic.unpack_seqdim(x, B)

    mag = torch.norm(xyz_velo0, dim=2)
    xyz_velo0 = xyz_velo0[:,mag[0]>1]
    xyz_velo0_bak = xyz_velo0.clone()

    intrins_ = __p(intrins)
    pix_T_cams_ = utils.geom.merge_intrinsics(*utils.geom.split_intrinsics(intrins_)).to(device)
    pix_T_cams = __u(pix_T_cams_)

    velo_T_cams = utils.geom.merge_rtlist(rots, trans).to(device)
    cams_T_velo = __u(utils.geom.safe_inverse(__p(velo_T_cams)))
    
    cam0_T_camXs = utils.geom.get_camM_T_camXs(velo_T_cams, ind=0)
    camXs_T_cam0 = __u(utils.geom.safe_inverse(__p(cam0_T_camXs)))
    cam0_T_camXs_ = __p(cam0_T_camXs)
    camXs_T_cam0_ = __p(camXs_T_cam0)
    
    xyz_cam0 = utils.geom.apply_4x4(cams_T_velo[:,0], xyz_velo0)
    rad_xyz_cam0 = utils.geom.apply_4x4(cams_T_velo[:,0], xyz_rad)

    lrtlist_cam0 = utils.geom.apply_4x4_to_lrtlist(cams_T_velo[:,0], lrtlist_velo)

    vox_util = utils.vox.Vox_util(
        Z, Y, X,
        scene_centroid=scene_centroid.to(device),
        bounds=bounds,
        assert_cube=False)
    
    V = xyz_velo0.shape[1]

    occ_mem0 = vox_util.voxelize_xyz(xyz_cam0, Z, Y, X, assert_cube=False)
    rad_occ_mem0 = vox_util.voxelize_xyz(rad_xyz_cam0, Z, Y, X, assert_cube=False)
    metarad_occ_mem0 = vox_util.voxelize_xyz_and_feats(rad_xyz_cam0, meta_rad, Z, Y, X, assert_cube=False)

    if not (model.module.use_radar or model.module.use_lidar):
        in_occ_mem0 = None
    elif model.module.use_lidar:
        assert(model.module.use_radar==False) # either lidar or radar, not both
        assert(model.module.use_metaradar==False) # either lidar or radar, not both
        in_occ_mem0 = occ_mem0
    elif model.module.use_radar and model.module.use_metaradar:
        in_occ_mem0 = metarad_occ_mem0
    elif model.module.use_radar:
        in_occ_mem0 = rad_occ_mem0
    elif model.module.use_metaradar:
        assert(False) # cannot use_metaradar without use_radar

    cam0_T_camXs = cam0_T_camXs

    lrtlist_cam0_g = lrtlist_cam0

    _, feat_bev_e, seg_bev_e, center_bev_e, offset_bev_e, _, height_e = model(
            rgb_camXs=rgb_camXs,
            pix_T_cams=pix_T_cams,
            cam0_T_camXs=cam0_T_camXs,
            vox_util=vox_util,
            rad_occ_mem0=in_occ_mem0)

    ce_loss = loss_fn(seg_bev_e, seg_bev_g, valid_bev_g)
    center_loss = balanced_mse_loss(center_bev_e, center_bev_g, valid_bev_g)
    offset_loss = torch.abs(offset_bev_e-offset_bev_g).sum(dim=1, keepdim=True)
    offset_loss = utils.basic.reduce_masked_mean(offset_loss, seg_bev_g*valid_bev_g)

    ce_factor = 1 / torch.exp(model.module.ce_weight)
    ce_loss = 10.0 * ce_loss * ce_factor
    ce_uncertainty_loss = 0.5 * model.module.ce_weight

    center_factor = 1 / (2*torch.exp(model.module.center_weight))
    center_loss = center_factor * center_loss
    center_uncertainty_loss = 0.5 * model.module.center_weight

    offset_factor = 1 / (2*torch.exp(model.module.offset_weight))
    offset_loss = offset_factor * offset_loss
    offset_uncertainty_loss = 0.5 * model.module.offset_weight

    # height loss
    if model.module.supervise_height:
        h,w = height_e.shape[-2], height_e.shape[-1]
        additional_bin = height_e[:, :, -1].reshape(-1, h, w)
        foreground_areas_reshaped = foreground_areas.reshape(-1, h, w)
        incomplete_objects_areas_reshaped = incomplete_objects_areas.reshape(-1, h, w)
        pix_masks_reshaped = pix_masks.reshape(-1, H, W)
        pix_masks_resized = F.interpolate(pix_masks_reshaped.unsqueeze(1), size=(h, w), mode='nearest').squeeze(1)

        bg_tg = 1 - foreground_areas_reshaped
        bg_mask = bg_tg * pix_masks_resized

        fg_tg = foreground_areas_reshaped
        fg_mask = fg_tg * pix_masks_resized * (1 - incomplete_objects_areas_reshaped)

        # for idx in range(bg_tg.shape[0]):
        #     if fg_mask[idx].min() > 0.5:
        #         bg_mask[idx] *= 0
        #         fg_mask[idx] *= 0
        #     if bg_mask[idx].sum() <= 0.:
        #         bg_mask[idx] *= 0

        height_loss = torch.topk(F.binary_cross_entropy(
                        additional_bin, bg_tg, reduction='none')[bg_mask > 0],
                        k=int(bg_mask.sum()*0.95), largest=False).values.mean() + \
                      torch.topk(F.binary_cross_entropy(
                        1-additional_bin, fg_tg, reduction='none')[fg_mask > 0],
                        k=int(fg_mask.sum()*0.7), largest=False).values.mean()
        # print('height_loss', height_loss)
        # height_loss = 0.0
        
        height_factor = 1 / (2*torch.exp(model.module.height_weight))
        height_loss = height_loss * height_factor
        height_uncertainty_loss = 0.5 * model.module.height_weight
        total_loss += height_loss
        total_loss += height_uncertainty_loss
        metrics['height_loss'] = height_loss.item()
        metrics['height_weight'] = model.module.height_weight.item()
        
    total_loss += ce_loss
    total_loss += center_loss
    total_loss += offset_loss
    total_loss += ce_uncertainty_loss
    total_loss += center_uncertainty_loss
    total_loss += offset_uncertainty_loss

    seg_bev_e_round = torch.sigmoid(seg_bev_e).round()
    intersection = (seg_bev_e_round*seg_bev_g*valid_bev_g).sum(dim=[1,2,3])
    union = ((seg_bev_e_round+seg_bev_g)*valid_bev_g).clamp(0,1).sum(dim=[1,2,3])
    iou = (intersection/(1e-4 + union)).mean()

    metrics['ce_loss'] = ce_loss.item()
    metrics['center_loss'] = center_loss.item()
    metrics['offset_loss'] = offset_loss.item()
    metrics['ce_weight'] = model.module.ce_weight.item()
    metrics['center_weight'] = model.module.center_weight.item()
    metrics['offset_weight'] = model.module.offset_weight.item()
    metrics['intersection'] = intersection.sum().item()
    metrics['union'] = union.sum().item()
    metrics['iou'] = iou.item()

    if sw is not None and sw.save_this:
        if model.module.use_radar or model.module.use_lidar:
            sw.summ_occ('0_inputs/rad_occ_mem0', rad_occ_mem0)
        sw.summ_occ('0_inputs/occ_mem0', occ_mem0)
        sw.summ_rgb('0_inputs/rgb_camXs', torch.cat(rgb_camXs[0:1].unbind(1), dim=-1))

        sw.summ_oned('2_outputs/feat_bev_e', torch.mean(feat_bev_e, dim=1, keepdim=True))
        # sw.summ_oned('2_outputs/feat_bev_e', torch.max(feat_bev_e, dim=1, keepdim=True)[0])
        
        sw.summ_oned('2_outputs/seg_bev_g', seg_bev_g * (0.5+valid_bev_g*0.5), norm=False)
        sw.summ_oned('2_outputs/valid_bev_g', valid_bev_g, norm=False)
        sw.summ_oned('2_outputs/seg_bev_e', torch.sigmoid(seg_bev_e).round(), norm=False, frame_id=iou.item())
        sw.summ_oned('2_outputs/seg_bev_e_soft', torch.sigmoid(seg_bev_e), norm=False)

        sw.summ_oned('2_outputs/center_bev_g', center_bev_g, norm=False)
        sw.summ_oned('2_outputs/center_bev_e', center_bev_e, norm=False)

        sw.summ_flow('2_outputs/offset_bev_e', offset_bev_e, clip=10)
        sw.summ_flow('2_outputs/offset_bev_g', offset_bev_g, clip=10)

        # pix maxk pixels
        sw.summ_oned('0_inputs/pix_masks', torch.cat(pix_masks[0:1].unbind(1), dim=-1), norm=False)

        # height_e
        if model.module.supervise_height:
            normal_grayscale = utils.improc.depth_to_grayscale(height_e[0])
            # save_image(normal_grayscale, 'normal_grayscale.jpg', nrow=normal_grayscale.shape[0], normalize=False)
            sw.summ_rgb('2_outputs/height_e', torch.cat(normal_grayscale.unsqueeze(0).unbind(1), dim=-1) - 0.5)

            sw.summ_oned('0_inputs/foreground_areas', torch.cat(foreground_areas[0:1].unbind(1), dim=-1), norm=False)
            sw.summ_oned('0_inputs/incomplete_objects_areas', torch.cat(incomplete_objects_areas[0:1].unbind(1), dim=-1), norm=False)
    return total_loss, metrics
    
def main(
        exp_name='debug',
        # training
        max_iters=100000,
        log_freq=1000,
        shuffle=True,
        dset='trainval',
        do_val=True,
        val_freq=100,
        save_freq=1000,
        batch_size=8,
        grad_acc=5,
        lr=3e-4,
        use_scheduler=True,
        weight_decay=1e-7,
        nworkers=12,
        # data/log/save/load directories
        data_dir='../nuscenes/',
        log_dir='logs_nuscenes_bevseg',
        ckpt_dir='checkpoints/',
        keep_latest=1,
        init_dir='',
        ignore_load=None,
        load_step=False,
        load_optimizer=False,
        # data
        res_scale=2,
        rand_flip=True,
        rand_crop_and_resize=True,
        ncams=6,
        nsweeps=3,
        data_ratio=1.0,
        # model
        encoder_type='res101',
        use_height_aware=True,
        supervise_height=False,
        use_radar=False,
        use_radar_filters=False,
        use_lidar=False,
        use_metaradar=False,
        do_rgbcompress=True,
        do_shuffle_cams=True,
        # cuda
        device_ids=[0,1,2,3],
        # debug
        val_all=False,
    ):

    kwargs = locals()
    kwargs.pop('kwargs', None)
    print("Keyword args:")
    for key, value in kwargs.copy().items():
        print(f"\t{key}: {value}")

    B = batch_size
    assert(B % len(device_ids) == 0) # batch size must be divisible by number of gpus
    if grad_acc > 1:
        print('effective batch size:', B*grad_acc)
    device = 'cuda:%d' % device_ids[0]

    # autogen a name
    model_name = "%d" % B
    if grad_acc > 1:
        model_name += "x%d" % grad_acc
    lrn = "%.1e" % lr # e.g., 5.0e-04
    lrn = lrn[0] + lrn[3:5] + lrn[-1] # e.g., 5e-4
    model_name += "_%s" % lrn
    if use_scheduler:
        model_name += "s"
    model_name += "_%s" % exp_name 
    import datetime
    model_date = datetime.datetime.now().strftime('%H_%M_%S')
    model_name = model_name + '_' + model_date
    print('model_name', model_name)

    # set up ckpt and logging 
    ckpt_dir = os.path.join(ckpt_dir, model_name)
    writer_t = SummaryWriter(os.path.join(log_dir, model_name + '/t'), max_queue=10, flush_secs=60)
    if do_val:
        writer_v = SummaryWriter(os.path.join(log_dir, model_name + '/v'), max_queue=10, flush_secs=60)

    # set up dataloaders
    final_dim = (int(224 * res_scale), int(400 * res_scale))
    print('resolution:', final_dim)

    if rand_crop_and_resize:
        resize_lim = [0.8,1.2]
        crop_offset = int(final_dim[0]*(1-resize_lim[0]))
    else:
        resize_lim = [1.0,1.0]
        crop_offset = 0
    
    data_aug_conf = {
        'crop_offset': crop_offset,
        'resize_lim': resize_lim,
        'final_dim': final_dim,
        'H': 900, 'W': 1600,
        'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
        'ncams': ncams,
    }
    train_dataloader, val_dataloader = nuscenesdataset.compile_data(
        dset,
        data_dir,
        data_aug_conf=data_aug_conf,
        centroid=scene_centroid_py,
        bounds=bounds,
        res_3d=(Z,Y,X),
        bsz=B,
        nworkers=nworkers,
        nworkers_val=0 if nworkers==0 else int(nworkers/2),
        shuffle=shuffle,
        use_radar_filters=use_radar_filters,
        seqlen=1, # we do not load a temporal sequence here, but that can work with this dataloader
        nsweeps=nsweeps,
        do_shuffle_cams=do_shuffle_cams,
        get_tids=True,
        data_ratio=data_ratio,
        supervise_height=supervise_height,
    )
    train_iterloader = iter(train_dataloader)
    val_iterloader = iter(val_dataloader)

    vox_util = utils.vox.Vox_util(
        Z, Y, X,
        scene_centroid=scene_centroid.to(device),
        bounds=bounds,
        assert_cube=False)

    # set up model & seg loss
    seg_loss_fn = SimpleLoss(2.13).to(device) # value from lift-splat
    model = Segnet(
        Z, Y, X, vox_util, 
        use_radar=use_radar, 
        use_lidar=use_lidar, 
        use_metaradar=use_metaradar, 
        do_rgbcompress=do_rgbcompress, 
        encoder_type=encoder_type, 
        rand_flip=rand_flip, 
        use_height_aware=use_height_aware,
        supervise_height=supervise_height)
    model = model.to(device)
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    parameters = list(model.parameters())
    if use_scheduler:
        optimizer, scheduler = fetch_optimizer(lr, weight_decay, 1e-8, max_iters, model.parameters())
    else:
        optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total_params', total_params)

    # load checkpoint
    global_step = 0
    if init_dir:
        if load_step and load_optimizer:
            global_step = saverloader.load(init_dir, model.module, optimizer, ignore_load=ignore_load)
        elif load_step:
            global_step = saverloader.load(init_dir, model.module, ignore_load=ignore_load)
        else:
            _ = saverloader.load(init_dir, model.module, ignore_load=ignore_load)
            global_step = 0
    requires_grad(parameters, True)
    model.train()

    # set up running logging pools
    n_pool = 10
    loss_pool_t = utils.misc.SimplePool(n_pool, version='np')
    time_pool_t = utils.misc.SimplePool(n_pool, version='np')
    iou_pool_t = utils.misc.SimplePool(n_pool, version='np')
    ce_pool_t = utils.misc.SimplePool(n_pool, version='np')
    center_pool_t = utils.misc.SimplePool(n_pool, version='np')
    offset_pool_t = utils.misc.SimplePool(n_pool, version='np')
    ce_weight_pool_t = utils.misc.SimplePool(n_pool, version='np')
    center_weight_pool_t = utils.misc.SimplePool(n_pool, version='np')
    offset_weight_pool_t = utils.misc.SimplePool(n_pool, version='np')
    if supervise_height:
        height_pool_t = utils.misc.SimplePool(n_pool, version='np')
        height_weight_pool_t = utils.misc.SimplePool(n_pool, version='np')
    if do_val:
        loss_pool_v = utils.misc.SimplePool(n_pool, version='np')
        iou_pool_v = utils.misc.SimplePool(n_pool, version='np')
        ce_pool_v = utils.misc.SimplePool(n_pool, version='np')
        center_pool_v = utils.misc.SimplePool(n_pool, version='np')
        offset_pool_v = utils.misc.SimplePool(n_pool, version='np')
        if supervise_height:
            height_pool_v = utils.misc.SimplePool(n_pool, version='np')
        if val_all:
            mean_iou_pool_v = utils.misc.SimplePool(n_pool, version='np')

    # training loop
    while global_step < max_iters:
        global_step += 1

        iter_start_time = time.time()
        iter_read_time = 0.0
        
        for internal_step in range(grad_acc):
            # read sample
            read_start_time = time.time()

            if internal_step==grad_acc-1:
                sw_t = utils.improc.Summ_writer(
                    writer=writer_t,
                    global_step=global_step,
                    log_freq=log_freq,
                    fps=2,
                    scalar_freq=int(log_freq/2),
                    just_gif=True)
            else:
                sw_t = None

            try:
                sample = next(train_iterloader)
            except StopIteration:
                train_iterloader = iter(train_dataloader)
                sample = next(train_iterloader)

            read_time = time.time()-read_start_time
            iter_read_time += read_time

            # run training iteration

            total_loss, metrics = run_model(model, seg_loss_fn, sample, device, sw_t)

            total_loss.backward()
        
        # if global_step % grad_acc == 0:
        torch.nn.utils.clip_grad_norm_(parameters, 5.0)
        optimizer.step()
        if use_scheduler:
            scheduler.step()
        optimizer.zero_grad()

        # update logging pools
        loss_pool_t.update([total_loss.item()])
        sw_t.summ_scalar('pooled/total_loss', loss_pool_t.mean())
        sw_t.summ_scalar('stats/total_loss', total_loss.item())

        iou_pool_t.update([metrics['iou']])
        sw_t.summ_scalar('pooled/iou', iou_pool_t.mean())
        sw_t.summ_scalar('stats/iou', metrics['iou'])

        ce_pool_t.update([metrics['ce_loss']])
        sw_t.summ_scalar('pooled/ce_loss', ce_pool_t.mean())
        sw_t.summ_scalar('stats/ce_loss', metrics['ce_loss'])
        
        ce_weight_pool_t.update([metrics['ce_weight']])
        sw_t.summ_scalar('pooled/ce_weight', ce_weight_pool_t.mean())
        sw_t.summ_scalar('stats/ce_weight', metrics['ce_weight'])
        
        center_pool_t.update([metrics['center_loss']])
        sw_t.summ_scalar('pooled/center_loss', center_pool_t.mean())
        sw_t.summ_scalar('stats/center_loss', metrics['center_loss'])

        center_weight_pool_t.update([metrics['center_weight']])
        sw_t.summ_scalar('pooled/center_weight', center_weight_pool_t.mean())
        sw_t.summ_scalar('stats/center_weight', metrics['center_weight'])
        
        offset_pool_t.update([metrics['offset_loss']])
        sw_t.summ_scalar('pooled/offset_loss', offset_pool_t.mean())
        sw_t.summ_scalar('stats/offset_loss', metrics['offset_loss'])

        offset_weight_pool_t.update([metrics['offset_weight']])
        sw_t.summ_scalar('pooled/offset_weight', offset_weight_pool_t.mean())
        sw_t.summ_scalar('stats/offset_weight', metrics['offset_weight'])

        if supervise_height:
            height_pool_t.update([metrics['height_loss']])
            sw_t.summ_scalar('pooled/height_loss', height_pool_t.mean())
            sw_t.summ_scalar('stats/height_loss', metrics['height_loss'])

            height_weight_pool_t.update([metrics['height_weight']])
            sw_t.summ_scalar('pooled/height_weight', height_weight_pool_t.mean())
            sw_t.summ_scalar('stats/height_weight', metrics['height_weight'])

        # run val
        if do_val and (global_step) % val_freq == 0:
            torch.cuda.empty_cache()
            model.eval()
            sw_v = utils.improc.Summ_writer(
                writer=writer_v,
                global_step=global_step,
                log_freq=log_freq,
                fps=5,
                scalar_freq=int(log_freq/2),
                just_gif=True)
            try:
                sample = next(val_iterloader)
            except StopIteration:
                val_iterloader = iter(val_dataloader)
                sample = next(val_iterloader)
                
            with torch.no_grad():
                total_loss, metrics = run_model(model, seg_loss_fn, sample, device, sw_v)

            # update val running pools
            loss_pool_v.update([total_loss.item()])
            sw_v.summ_scalar('pooled/total_loss', loss_pool_v.mean())
            sw_v.summ_scalar('stats/total_loss', total_loss.item())

            iou_pool_v.update([metrics['iou']])
            sw_v.summ_scalar('pooled/iou', iou_pool_v.mean())

            ce_pool_v.update([metrics['ce_loss']])
            sw_v.summ_scalar('pooled/ce_loss', ce_pool_v.mean())

            center_pool_v.update([metrics['center_loss']])
            sw_v.summ_scalar('pooled/center_loss', center_pool_v.mean())

            offset_pool_v.update([metrics['offset_loss']])
            sw_v.summ_scalar('pooled/offset_loss', offset_pool_v.mean())

            if supervise_height:
                height_pool_v.update([metrics['height_loss']])
                sw_v.summ_scalar('pooled/height_loss', height_pool_v.mean())

            model.train()

        # val all
        mean_iou = 0
        if do_val and val_all and np.mod(global_step, save_freq)==0:
            torch.cuda.empty_cache()
            model.eval()
            val_all_iterloader = iter(val_dataloader)
            val_step = 0
            intersection = 0
            union = 0
            max_iters_val = len(val_dataloader)
            while val_step < max_iters_val:
                val_step += 1
                try:
                    sample = next(val_all_iterloader)
                except StopIteration:
                    break
                with torch.no_grad():
                    total_loss, metrics = run_model(model, seg_loss_fn, sample, device, None)
                intersection += metrics['intersection']
                union += metrics['union']
            mean_iou = intersection/union
            print('%s; setp %06d/%d; mean_iou %.2f' % (model_name, global_step, max_iters, 100*mean_iou))
            mean_iou_pool_v.update([mean_iou])
            sw_v.summ_scalar('pooled/mean_iou', mean_iou)
            model.train()
        
        # save model checkpoint
        if np.mod(global_step, save_freq)==0:
            saverloader.save(ckpt_dir, optimizer, model.module, global_step, keep_latest=keep_latest)
        
        # log lr and time
        current_lr = optimizer.param_groups[0]['lr']
        sw_t.summ_scalar('_/current_lr', current_lr)
        
        iter_time = time.time()-iter_start_time
        time_pool_t.update([iter_time])
        sw_t.summ_scalar('pooled/time_per_batch', time_pool_t.mean())
        sw_t.summ_scalar('pooled/time_per_el', time_pool_t.mean()/float(B))

        if do_val:
            print('%s; step %06d/%d; rtime %.2f; itime %.2f; loss %.5f; iou_t %.1f; iou_v %.1f' % (
                model_name, global_step, max_iters, iter_read_time, iter_time,
                total_loss.item(), 100*iou_pool_t.mean(), 100*iou_pool_v.mean()))
        else:
            print('%s; step %06d/%d; rtime %.2f; itime %.2f; loss %.5f; iou_t %.1f' % (
                model_name, global_step, max_iters, iter_read_time, iter_time,
                total_loss.item(), 100*iou_pool_t.mean()))
            
    writer_t.close()
    if do_val:
        writer_v.close()
            

if __name__ == '__main__':
    Fire(main)


