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
from model.gan_network import Generator, Discriminator
from model.gradient_penalty import GradientPenalty
from evaluation.FPD import calculate_fpd

from metrics import *

from evaluation.pointnet import PointNetCls

# from arguments import Arguments
import argparse
import time
import visdom
import numpy as np
import time
import os.path as osp
import os

def EMD():
    raise NotImplementedError("Not implemented yet!!!")

def CD():
    raise NotImplementedError("Not implemented yet!!!")

def JSD():
    raise NotImplementedError("Not implemented yet!!!")

def MMD():
    raise NotImplementedError("Not implemented yet!!!")

def Coverage():
    raise NotImplementedError("Not implemented yet!!!")

def save_pointcloud_to_txt(batch_numpy,save_dir):
    batch_size = batch_numpy.shape[0]
    if not osp.isdir(save_dir):
        os.mkdir(save_dir)
    for i in range(batch_size):
       np.savetxt(osp.join(save_dir,str(i)+'.txt'), batch_numpy[i], fmt = "%f;%f;%f")  





parser = argparse.ArgumentParser()

parser.add_argument('--dataset_path', type=str, default='/mnt/lustre/share/zhangjunzhe/shapenetcore_partanno_segmentation_benchmark_v0', help='Dataset file path.')

parser.add_argument('--num_samples',type=int, default=5000, help='number for points to be generated by the G')
parser.add_argument('--model_pathname', default='./model/checkpoints18/tree_ckpt_1660_Chair.pt',help='pathname of the GAN to be evaled')
# parser.add_argument('--model_pathname', default='./model/checkpoints18/tree_ckpt_1430_Airplane.pt',help='pathname of the GAN to be evaled')
parser.add_argument('--FPD_path', type=str, default='./evaluation/pre_statistics_chair.npz', help='Statistics file path to evaluate FPD metric. (default:all_class)')
parser.add_argument('--class_choice', type=str, default='Chair', help='Select one class to generate. [Airplane, Chair, ...] (default:all_class)')
parser.add_argument('--support', type=int, default=10, help='Support value for TreeGCN loop term.')
parser.add_argument('--DEGREE', type=int, default=[1,  2,   2,   2,   2,   2,   64], nargs='+', help='Upsample degrees for generator.')
parser.add_argument('--G_FEAT', type=int, default=[96, 256, 256, 256, 128, 128, 128, 3], nargs='+', help='Features for generator.')
parser.add_argument('--batch_size', type=int, default=64, help='Integer value for batch size.')
parser.add_argument('--save_num_generated', type=int,default=100, help ='number of point clouds to be saved')
parser.add_argument('--save_generated_dir',required=True,help='dir to save generated point clouds')      
opt = parser.parse_args()
print(opt)

tic  = time.time()
device = torch.device('cuda')

opt.device = device

G = Generator(batch_size=opt.batch_size, features=opt.G_FEAT, degrees=opt.DEGREE, support=opt.support).to(device)
checkpoint = torch.load(opt.model_pathname, map_location=device)
G.load_state_dict(checkpoint['G_state_dict'])
G.eval()

fake_pointclouds = torch.Tensor([])
# jz, adjust for different batch size
test_batch_num = int(opt.num_samples/opt.batch_size)
print ('test_batch_num, num_samples, batch_size:', test_batch_num,opt.num_samples,opt.batch_size)
for i in range(test_batch_num): # For 5000 samples
    z = torch.randn(opt.batch_size, 1, 96).to(opt.device)
    tree = [z]
    with torch.no_grad():
        sample = G(tree).cpu()
    fake_pointclouds = torch.cat((fake_pointclouds, sample), dim=0)
print ('sample_pcs',fake_pointclouds.shape)
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
gt_dataset = BenchmarkDataset(root=opt.dataset_path, npoints=2048, uniform=None, class_choice=opt.class_choice)
dataLoader = torch.utils.data.DataLoader(gt_dataset, batch_size=opt.batch_size, shuffle=True, pin_memory=True, num_workers=10)
gt_data_list = []
for _iter, data in enumerate(dataLoader):
    point, _  = data
    gt_data_list.append(point)

ref_pcs = torch.cat(gt_data_list,0).detach().cpu().numpy()
sample_pcs = fake_pointclouds.detach().cpu().numpy()
# ref_pcs = torch.stack(gt_data_list).detach().cpu().numpy()
print ('shape of generated data and ref_pcs,',fake_pointclouds.shape, ref_pcs.shape)

# # jsd = jsd_between_point_cloud_sets(sample_pcs, ref_pcs, resolution=28)
# jsd1 = jsd_between_point_cloud_sets(sample_pcs, ref_pcs, resolution=28)
# jsd2 = jsd_between_point_cloud_sets(ref_pcs[:3000], ref_pcs, resolution=28)
# jsd3 = jsd_between_point_cloud_sets(sample_pcs[:2000], ref_pcs[-1000:], resolution=28)
# print ('jsd1, 2, 3', jsd1, jsd2, jsd3)
# # jsd1, 2, 3 0.11505738867010251 0.0002315239257608681 0.1166695618494149  


ae_loss = 'chamfer'  # Which distance to use for the matchings.
ae_loss = 'emd'  # Which distance to use for the matchings.

if ae_loss == 'emd':
    use_EMD = True
else:
    use_EMD = False  # Will use Chamfer instead.
    
batch_size = 100     # Find appropriate number that fits in GPU.
normalize = True     # Matched distances are divided by the number of 
                     # points of thepoint-clouds.

mmd, matched_dists = minimum_mathing_distance(sample_pcs[:20], ref_pcs[:20], batch_size, normalize=normalize, use_EMD=use_EMD)
print ('samp vs ref',mmd, matched_dists)
mmd, matched_dists = minimum_mathing_distance(ref_pcs[:15], ref_pcs[:20], batch_size, normalize=normalize, use_EMD=use_EMD)
print ('ref vs ref',mmd, matched_dists)
# MMD for 100 vs 100, 800s, very slow!!
# samp vs ref 0.0020890734 

# cov, matched_loc, matched_dist = coverage(sample_pcs[:100], ref_pcs[:100], batch_size, normalize=normalize, use_EMD=use_EMD)
# print ('samp vs ref',cov,matched_loc, matched_dist)
# # cov, matched_loc, matched_dist = coverage(ref_pcs[:18], ref_pcs[:20], batch_size, normalize=normalize, use_EMD=use_EMD)
# # print ('ref vs ref',cov,matched_loc, matched_dist)

del fake_pointclouds
toc = time.time()
print('time spent is',int(toc-tic))

# ####
# check point cloud tensor
# ####
# plane_dir = '/mnt/lustre/share/zhangjunzhe/shapenetcore_partanno_segmentation_benchmark_v0/02691156/points'

# plane_name = '6c8275f09052bf66ca8607f540cc62ba.pts'

# # # cp /mnt/lustre/share/zhangjunzhe/shapenetcore_partanno_segmentation_benchmark_v0/02691156/points/6c8275f09052bf66ca8607f540cc62ba.pts ~

# plane_pathname = osp.join(plane_dir,plane_name)

# point_set = np.loadtxt(plane_pathname).astype(np.float32)

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