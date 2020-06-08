"""
This file evaluate generated points from a GAN compare with 

It first generate point sets from a given GAN

It should be saving generated points

It should also be 

"""
import torch
import torch.nn as nn
import torch.optim as optim

from data.dataset_benchmark import BenchmarkDataset
from datasets import ShapeNet_v0
from model.gan_network import Generator, Discriminator
from train_cgan import ConditionalGenerator_v0
from model.gradient_penalty import GradientPenalty
from evaluation.FPD import calculate_fpd, calculate_activation_statistics

from metrics import *
from loss import *

from evaluation.pointnet import PointNetCls
from math import ceil
# from arguments import Arguments
import argparse
import time
import visdom
import numpy as np
import time
import os.path as osp
import os
# import Namespace
import copy

def count_shapenet_v0():
    root_dir = '/mnt/lustre/share/zhangjunzhe/shapenetcore_partanno_segmentation_benchmark_v0'
    catfile = './data/synsetoffset2category.txt'
    class2dir_dict = {}
    with open(catfile, 'r') as f:
        for line in f:
            ls = line.strip().split()
            class2dir_dict[ls[0]] = ls[1]
    class2id_dict = {'Airplane': 0, 'Bag': 1, 'Cap': 2, 'Car': 3, 'Chair': 4, 
    'Earphone': 5, 'Guitar': 6,  'Knife': 7, 'Lamp': 8, 'Laptop': 9, 
    'Motorbike': 10, 'Mug': 11, 'Pistol': 12, 'Rocket': 13, 'Skateboard': 14, 'Table': 15}   
    count_list = [0] * 16
    class2cnt_dict = {}
    print(class2dir_dict)
    print(class2id_dict)
    for class_name, id in class2id_dict.items():
        dir_point = os.path.join(root_dir, class2dir_dict[class_name], 'points')
        fns = os.listdir(dir_point)
        count_list[id] = len(fns)
        class2cnt_dict[class_name] = len(fns)
    print(class2cnt_dict)
    print(np.sum(count_list))
    return count_list


def save_pcs_to_txt(save_dir, fake_pcs, labels=None):
    sample_size = fake_pcs.shape[0]
    if not osp.isdir(save_dir):
        os.mkdir(save_dir)
    for i in range(sample_size):
        if labels is None:
            np.savetxt(osp.join(save_dir,str(i)+'.txt'), fake_pcs[i], fmt = "%f;%f;%f")  
        else:
            np.savetxt(osp.join(save_dir,str(i)+'_'+str(labels[i])+'.txt'), fake_pcs[i], fmt = "%f;%f;%f") 

def generate_pcs(model_cuda, n_pcs=5000, batch_size=64, n_classes=1,ratio=None, conditional=False,device=None):
    # import pdb; pdb.set_trace()
    fake_pcs = torch.Tensor([])
    all_gen_labels = torch.Tensor([])
    n_pcs = int(ceil(n_pcs/batch_size) * batch_size)
    n_batches = ceil(n_pcs/batch_size)
    if not conditional:
    # if n_classes == 1 or ratio is None:
        for i in range(n_batches):
            z = torch.randn(opt.batch_size, 1, 96).to(device)
            tree = [z]
            with torch.no_grad():
                sample = model_cuda(tree).cpu()
            fake_pcs = torch.cat((fake_pcs, sample), dim=0)
    elif conditional and ratio is False:
        for i in range(n_batches):
            # import pdb; pdb.set_trace()
            z = torch.randn(batch_size, 1, 96).to(device)
            gen_labels = torch.from_numpy(np.random.randint(0, opt.n_classes, batch_size).reshape(-1,1)).to(device)
            all_gen_labels = torch.cat((all_gen_labels,gen_labels.cpu()),0)
            gen_labels_onehot = torch.FloatTensor(batch_size, opt.n_classes).to(device)
            gen_labels_onehot.zero_()
            gen_labels_onehot.scatter_(1, gen_labels, 1)
            gen_labels_onehot.unsqueeze_(1)
            tree = [z]
            with torch.no_grad():
                sample = model_cuda(tree,gen_labels_onehot).cpu()
            fake_pcs = torch.cat((fake_pcs, sample), dim=0)
    else:
        # n_pcs = 300
        # NOTE: due to non-whole batch resulting issue in some models, round up n_pcs
        
        # got ratio, assume which is a list of counts from the training data
        # print (n_pcs)
        # NOTE, here shoulb change if n_classes changed
        # ratio = count_shapenet_v0()
        ratio = [500] * 4
        # print (ratio)
        ratio_nm = np.array(ratio)/np.sum(ratio)
        # print (ratio_nm)
        ratio_cnt = ratio_nm * n_pcs
        # just check all chair scenario 
        # ratio_cnt = [0] * 4 + [5056] + [0] * 11
        # print (ratio_cnt)
        all_gen_labels = torch.zeros(n_pcs).type(torch.LongTensor).reshape(-1,1)
        # NOTE due to some r is not int, there might be last a few all_gen_labels value remain at 0.
        # TODO to shuffle the tensor
        pointer = 0
        for i, r in enumerate(ratio_cnt):
            all_gen_labels[pointer:(pointer+int(r))] = int(i)
            pointer+=int(r)
        # print(all_gen_labels)
        for i in range(n_batches):
            # import pdb; pdb.set_trace()
            z = torch.randn(batch_size, 1, 96).to(device)
            gen_labels = all_gen_labels[i*batch_size:(i+1)*batch_size].reshape(-1,1).to(device)
            # print(gen_labels.dtype)
            gen_labels_onehot = torch.FloatTensor(batch_size, opt.n_classes).to(device)
            gen_labels_onehot.zero_()
            gen_labels_onehot.scatter_(1, gen_labels, 1)
            gen_labels_onehot.unsqueeze_(1)
            tree = [z]
            with torch.no_grad():
                sample = model_cuda(tree,gen_labels_onehot).cpu()
            fake_pcs = torch.cat((fake_pcs, sample), dim=0)

    return fake_pcs, all_gen_labels

def FPD(opt,save_gen=False):
    '''
    NOTE: model is of a certain class now
    args needed: 
        n_classes, pcs to generate, ratio of each class, class to id dict???
        model pth, , points to save, save pth, npz for the class, 
    '''
    # print(' in FPD')
    if not opt.conditional:
        G_net = Generator(batch_size=opt.batch_size, features=opt.G_FEAT, degrees=opt.DEGREE, support=opt.support,version=opt.version).to(device)
    else:
        G_net = ConditionalGenerator_v0(batch_size=opt.batch_size, features=opt.G_FEAT, degrees=opt.DEGREE, support=opt.support, n_classes=opt.n_classes,version=opt.version).to(opt.device)
    # print(G_net)
    # print(opt.model_pathname, opt.version)
    checkpoint = torch.load(opt.model_pathname, map_location=device)
    G_net.load_state_dict(checkpoint['G_state_dict'])
    G_net.eval()
    # compute ratio 
    # if not conditional, labels are dummy
    fake_pcs, labels = generate_pcs(G_net, n_pcs = opt.num_samples, batch_size = opt.batch_size, conditional=opt.conditional, device=opt.device, ratio = opt.conditional_ratio)
    # print('fake_pcs shape,',fake_pcs.shape)

    if save_gen:
        save_pcs_to_txt(opt.gen_path, fake_pcs, labels=labels)
    # TODO check all-chair only scenario
    # opt.FPD_path = './evaluation/pre_statistics_chair.npz'
    fpd = calculate_fpd(fake_pcs, statistic_save_path=opt.FPD_path, batch_size=100, dims=1808, device=opt.device)
    # print('Frechet Pointcloud Distance <<< {:.4f} >>>'.format(fpd))
    
    return fpd

def create_fpd_stats(pcs, pathname_save, device):
    # pcs = pcs.transpose(1,2)
    PointNet_pretrained_path = './evaluation/cls_model_39.pth'

    model = PointNetCls(k=16).to(device)
    model.load_state_dict(torch.load(PointNet_pretrained_path))
    mu, sigma = calculate_activation_statistics(pcs, model, device=device)
    print (mu.shape, sigma.shape)
    np.savez(pathname_save,m=mu,s=sigma)
    print('saved into', pathname_save)
    # f = np.load(pathname_save)
    # m2, s2 = f['m'][:], f['s'][:]
    # f.close()
    # print('loading, m2, s2 shape',m2.shape, s2.shape)

def script_create_fpd_stats(opt):
    pathname_save = './evaluation/pre_statistics_4x500.npz'
    class_choice = ['Airplane','Car','Chair','Table']
    ratio = [500] * 4
    dataset = ShapeNet_v0(root=opt.dataset_path, npoints=opt.point_num, uniform=None, class_choice=class_choice,ratio=ratio)
    dataLoader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, pin_memory=True, num_workers=8)
    ref_pcs = torch.Tensor([])
    for _iter, data in enumerate(dataLoader):
        point, labels = data
        ref_pcs = torch.cat((ref_pcs, point),0)
    print ('shape of ref_pcs',ref_pcs.shape)
    create_fpd_stats(ref_pcs,pathname_save, opt.device)

# def visualize_pcd_to_png()


parser = argparse.ArgumentParser()

parser.add_argument('--dataset_path', type=str, default='/mnt/lustre/share/zhangjunzhe/shapenetcore_partanno_segmentation_benchmark_v0', help='Dataset file path.')

parser.add_argument('--num_samples',type=int, default=5000, help='number for points to be generated by the G')
parser.add_argument('--model_pathname', default='./model/checkpoints18/tree_ckpt_1660_Chair.pt',help='pathname of the GAN to be evaled')
parser.add_argument('--model_path', default='./model/checkpoints18',help='pathname of the GAN to be evaled')
# parser.add_argument('--model_pathname', default='./model/checkpoints18/tree_ckpt_1430_Airplane.pt',help='pathname of the GAN to be evaled')
parser.add_argument('--FPD_path', type=str, default='./evaluation/pre_statistics_chair.npz', help='Statistics file path to evaluate FPD metric. (default:all_class)')
parser.add_argument('--class_choice', type=str, default='Chair', help='Select one class to generate. [Airplane, Chair, ...] (default:all_class)')
parser.add_argument('--support', type=int, default=10, help='Support value for TreeGCN loop term.')
parser.add_argument('--DEGREE', type=int, default=[1,  2,   2,   2,   2,   2,   64], nargs='+', help='Upsample degrees for generator.')
parser.add_argument('--G_FEAT', type=int, default=[96, 256, 256, 256, 128, 128, 128, 3], nargs='+', help='Features for generator.')
parser.add_argument('--batch_size', type=int, default=64, help='Integer value for batch size.')
parser.add_argument('--save_num_generated', type=int,default=100, help ='number of point clouds to be saved')
parser.add_argument('--gen_path',required=True,help='dir to save generated point clouds')   
# parser.add_argument('--conditional',type=int, required=True, help='1 is conditional , 0 is false')   
parser.add_argument('--conditional', default=False, type=lambda x: (str(x).lower() == 'true'))  
parser.add_argument('--n_classes',type=int, default=16)
parser.add_argument('--conditional_ratio',default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--point_num', type=int, default=2048, help='Integer value for number of points.')
parser.add_argument('--version', type=int, default=0)
parser.add_argument('--save_gen',default=False, type=lambda x: (str(x).lower() == 'true'))

opt = parser.parse_args()
device = torch.device('cuda')
opt.device = device
print(opt)
print(opt.conditional,type(opt.conditional))
print(opt.conditional_ratio,type(opt.conditional_ratio))
tic  = time.time()

################################# above is common section


######################## fpd conditional
# epoch_checkpoints = [1500]
# # epoch_checkpoints = range(2000, 0, -100)
# for epoch in epoch_checkpoints:
#     if epoch == 2000:
#         epoch = 1995
#     opt_deep_copy = copy.deepcopy(opt)
#     tic = time.time()
#     opt_deep_copy.model_pathname = opt_deep_copy.model_path + '/tree_ckpt_'+str(epoch)+'_None.pt'
#     fpd_value = FPD(opt_deep_copy,save_gen=opt_deep_copy.save_gen)
#     toc = time.time()
#     print ('--------------------time spent:',int(toc-tic),'|| epoch:',epoch,'|| FPD: <<< {:.2f} >>>'.format(fpd_value))



######################## generate pcs and save 
# srun -u --partition=Sensetime -n1 --gres=gpu:1 --ntasks-per-node=1 -x SG-IDC1-10-51-0-30 \
#         --job-name=eval_g \
#         python eval_GAN.py \
#         --FPD_path ./evaluation/pre_statistics_chair.npz \
#         --class_choice Chair \
#         --save_num_generated 500 \
#         --gen_path ./gen_temp \
#         --model_pathname ./model/author_checkpoint/tree_ckpt_Airplane.pt 

# G = Generator(batch_size=opt.batch_size, features=opt.G_FEAT, degrees=opt.DEGREE, support=opt.support).to(opt.device)
# if not opt.conditional:
#     G_net = Generator(batch_size=opt.batch_size, features=opt.G_FEAT, degrees=opt.DEGREE, support=opt.support,version=opt.version).to(device)
# else:
#     G_net = ConditionalGenerator_v0(batch_size=opt.batch_size, features=opt.G_FEAT, degrees=opt.DEGREE, support=opt.support, n_classes=opt.n_classes,version=opt.version).to(opt.device)
# checkpoint = torch.load(opt.model_pathname, map_location=device)
# G_net.load_state_dict(checkpoint['G_state_dict'])
# G_net.eval()
# # import pdb; pdb.set_trace()
# fake_pcs, _ = generate_pcs(G_net, 100, 100, device= opt.device)
# fake_pcs = fake_pcs.detach().cpu().numpy()
# # import pdb; pdb.set_trace()
# save_pcs_to_txt('./gen_temp',fake_pcs)

# fake_pointclouds = torch.Tensor([])
# # jz, adjust for different batch size
# test_batch_num = int(opt.num_samples/opt.batch_size)
# print ('test_batch_num, num_samples, batch_size:', test_batch_num,opt.num_samples,opt.batch_size)
# for i in range(test_batch_num): # For 5000 samples
#     z = torch.randn(opt.batch_size, 1, 96).to(opt.device)
#     tree = [z]
#     with torch.no_grad():
#         sample = G(tree).cpu()
#     fake_pointclouds = torch.cat((fake_pointclouds, sample), dim=0)
# print ('sample_pcs',fake_pointclouds.shape)
### fpd
# import pdb; pdb.set_trace()
# fpd = calculate_fpd(fake_pointclouds, statistic_save_path=opt.FPD_path, batch_size=100, dims=1808, device=opt.device)
# # # metric['FPD'].append(fpd)
# print('Frechet Pointcloud Distance <<< {:.4f} >>>'.format(fpd))

### Section save point cloud
# pointclouds_to_save = fake_pointclouds.numpy()[:opt.save_num_generated]
# save_pointcloud_to_txt(pointclouds_to_save,opt.save_generated_dir)
# print ('saved point clouds')

###
# get point clouds for a particular data
# gt_dataset = BenchmarkDataset(root=opt.dataset_path, npoints=2048, uniform=None, class_choice=opt.class_choice)
# dataLoader = torch.utils.data.DataLoader(gt_dataset, batch_size=opt.batch_size, shuffle=True, pin_memory=True, num_workers=10)
# gt_data_list = []
# for _iter, data in enumerate(dataLoader):
#     point, _  = data
#     gt_data_list.append(point)

# ref_pcs = torch.cat(gt_data_list,0).detach().cpu().numpy()
# sample_pcs = fake_pointclouds.detach().cpu().numpy()
# # ref_pcs = torch.stack(gt_data_list).detach().cpu().numpy()
# print ('shape of generated data and ref_pcs,',fake_pointclouds.shape, ref_pcs.shape)

# # # jsd = jsd_between_point_cloud_sets(sample_pcs, ref_pcs, resolution=28)
# # jsd1 = jsd_between_point_cloud_sets(sample_pcs, ref_pcs, resolution=28)
# # jsd2 = jsd_between_point_cloud_sets(ref_pcs[:3000], ref_pcs, resolution=28)
# # jsd3 = jsd_between_point_cloud_sets(sample_pcs[:2000], ref_pcs[-1000:], resolution=28)
# # print ('jsd1, 2, 3', jsd1, jsd2, jsd3)
# # # jsd1, 2, 3 0.11505738867010251 0.0002315239257608681 0.1166695618494149  


# ae_loss = 'chamfer'  # Which distance to use for the matchings.
# ae_loss = 'emd'  # Which distance to use for the matchings.

# if ae_loss == 'emd':
#     use_EMD = True
# else:
#     use_EMD = False  # Will use Chamfer instead.
    
# batch_size = 100     # Find appropriate number that fits in GPU.
# normalize = True     # Matched distances are divided by the number of 
#                      # points of thepoint-clouds.

# mmd, matched_dists = minimum_mathing_distance(sample_pcs[:20], ref_pcs[:20], batch_size, normalize=normalize, use_EMD=use_EMD)
# print ('samp vs ref',mmd, matched_dists)
# mmd, matched_dists = minimum_mathing_distance(ref_pcs[:15], ref_pcs[:20], batch_size, normalize=normalize, use_EMD=use_EMD)
# print ('ref vs ref',mmd, matched_dists)
# MMD for 100 vs 100, 800s, very slow!!
# samp vs ref 0.0020890734 

# cov, matched_loc, matched_dist = coverage(sample_pcs[:100], ref_pcs[:100], batch_size, normalize=normalize, use_EMD=use_EMD)
# print ('samp vs ref',cov,matched_loc, matched_dist)
# # cov, matched_loc, matched_dist = coverage(ref_pcs[:18], ref_pcs[:20], batch_size, normalize=normalize, use_EMD=use_EMD)
# # print ('ref vs ref',cov,matched_loc, matched_dist)

# del fake_pointclouds
# toc = time.time()
# print('time spent is',int(toc-tic))

# ####
# check point cloud tensor
# ####
plane_dir = '/mnt/lustre/share/zhangjunzhe/shapenetcore_partanno_segmentation_benchmark_v0/02691156/points'

plane_name = '6c8275f09052bf66ca8607f540cc62ba.pts'

# # # cp /mnt/lustre/share/zhangjunzhe/shapenetcore_partanno_segmentation_benchmark_v0/02691156/points/6c8275f09052bf66ca8607f540cc62ba.pts ~

plane_pathname = osp.join(plane_dir,plane_name)

point_set = np.loadtxt(plane_pathname).astype(np.float32)
print('pc shape',point_set.shape)
pc_t = torch.from_numpy(point_set)
pcs = torch.stack([pc_t]*64)
print('pcs shape',pcs.shape)
seeds = farthest_point_sample(pcs,10) # returned index, not coordinates.
print('seeds shape',seeds.shape)
print(seeds[0])
seed_points = point_set[seeds[0]]
print(seed_points)
patches = extract_knn_patch(seed_points,point_set,50)
print(patches[0].shape)
print('===================')
# print(patches)
radius = 0.005
nsample = 500
new_xyz = torch.stack([pc[seed] for pc, seed in zip(pcs,seeds)])
print('new_xyz shape',new_xyz.shape)

group_idx = query_ball_point(radius, nsample, pcs, new_xyz)
# print ('group idx shape')

# point_set_tensor = torch.from_numpy(point_set)

# point_set_2 = point_set_tensor.numpy()
# print(point_set.shape,point_set_tensor.shape,point_set_2.shape)

# np.savetxt('sample_pcd.txt', point_set_2, fmt = "%f,%f,%f") 

####
# check classification
###
# PointNet_pretrained_path = './evaluation/cls_model_39.pth'
# model = PointNetCls(k=16)
# model.load_state_dict(torch.load(PointNet_pretrained_path))
# model.to(device)
# # fake_pointclouds = fake_pointclouds.transpose(1,2)
# # soft, trans, actv = model(fake_pointclouds.to(device))
# # import pdb; pdb.set_trace()

# dataset = BenchmarkDataset(root=opt.dataset_path, npoints=2048, uniform=None, class_choice=opt.class_choice)
# dataLoader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=0)

# for _iter, data in enumerate(dataLoader):
#     point, _ = data
#     point = point.transpose(1,2).to(opt.device)
#     soft, trans, actv = model(point)
#     import pdb; pdb.set_trace()