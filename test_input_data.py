"""
The goal of this file is to:
* check number of points in the raw input
* visualize the sampled points, say 100

"""

from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
from arguments import Arguments

class BenchmarkDataset(data.Dataset):
    #jz default classification=False
    def __init__(self, root, npoints=2500, uniform=False, classification=True, class_choice=None):
        self.npoints = npoints
        self.root = root
        self.catfile = './data/synsetoffset2category.txt'
        self.cat = {}
        self.num_points_list =[] # number of points in the point
        self.uniform = uniform
        self.classification = classification

        
        #jz: if None, just all cat; if not None, just a single cat
        # with open(self.catfile, 'r') as f:
        #     for line in f:
        #         ls = line.strip().split()
        #         self.cat[ls[0]] = ls[1]
        # import pdb; pdb.set_trace()

        #jz input 'None' is str
        if class_choice != 'None':
            self.cat = {k:v for k,v in self.cat.items() if k in class_choice}
        
        print ('self.cat:',self.cat)

        self.meta = {}
        
        for item in self.cat:
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item], 'points')
            dir_seg = os.path.join(self.root, self.cat[item], 'points_label')
            dir_sampling = os.path.join(self.root, self.cat[item], 'sampling')

            fns = sorted(os.listdir(dir_point))

            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append((os.path.join(dir_point, token + '.pts'), os.path.join(dir_seg, token + '.seg'), os.path.join(dir_sampling, token + '.sam')))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn[0], fn[1], fn[2]))
        # import pdb; pdb.set_trace()

        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))
        print(self.classes)
        self.num_seg_classes = 0
        if not self.classification:
            for i in range(len(self.datapath)//50):
                l = len(np.unique(np.loadtxt(self.datapath[i][-1]).astype(np.uint8)))
                if l > self.num_seg_classes:
                    self.num_seg_classes = l

    # def get_num_points(self, index):

    def __getitem__(self, index):
        fn = self.datapath[index]
        cls = self.classes[self.datapath[index][0]]
        point_set = np.loadtxt(fn[1]).astype(np.float32)
        seg = np.loadtxt(fn[2]).astype(np.int64)
        self.num_points_list.append(point_set.shape[0])
        if self.uniform:
            choice = np.loadtxt(fn[3]).astype(np.int64)
            assert len(choice) == self.npoints, "Need to match number of choice(2048) with number of vertices."
        else:
            choice = np.random.randint(0, len(seg), size=self.npoints)

        point_set = point_set[choice]
        seg = seg[choice]

        point_set = torch.from_numpy(point_set)
        seg = torch.from_numpy(seg)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))
        if self.classification:
            return point_set, cls
        else:
            return point_set, seg

    def __len__(self):
        return len(self.datapath)



### test scrip
print('start runing')
args = Arguments().parser().parse_args()

args.device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(args.device)
data = BenchmarkDataset(root=args.dataset_path, npoints=args.point_num, uniform=None, class_choice='Chair')
for i in range(10):
    data.__getitem__()
print(len(data.num_points_list))
print(data.num_points_list)
