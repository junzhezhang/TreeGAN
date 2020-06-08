import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
### 
def square_distance(src, dst):
    # link: http://www.programmersought.com/article/8737853003/#14_query_ball_point_93
    """
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2     
	     = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1)) # 2*(xn * xm + yn * ym + zn * zm)
    dist += torch.sum(src ** 2, -1).view(B, N, 1) # xn*xn + yn*yn + zn*zn
    dist += torch.sum(dst ** 2, -1).view(B, 1, M) # xm*xm + ym*ym + zm*zm
    return dist
def farthest_point_sample(xyz, npoint):
    # link: http://www.programmersought.com/article/8737853003/#14_query_ball_point_93
    # form: torch tensor
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
    	# Update the i-th farthest point
        centroids[:, i] = farthest
        # Take the xyz coordinate of the farthest point
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        # Calculate the Euclidean distance from all points in the point set to this farthest point
        dist = torch.sum((xyz - centroid) ** 2, -1)
        # Update distances to record the minimum distance of each point in the sample from all existing sample points
        mask = dist < distance
        distance[mask] = dist[mask]
        # Find the farthest point from the updated distances matrix, and use it as the farthest point for the next iteration
        farthest = torch.max(distance, -1)[1]
    return centroids

def query_ball_point(radius, nsample, xyz, new_xyz):
    # link: http://www.programmersought.com/article/8737853003/#14_query_ball_point_93    
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    # sqrdists: [B, S, N] Record the Euclidean distance between the center point and all points
    sqrdists = square_distance(new_xyz, xyz)
    # import pdb; pdb.set_trace()
    # Find all distances greater than radius^2, its group_idx is directly set to N; the rest retain the original value
    group_idx[sqrdists > radius ** 2] = N
    # Do ascending order, the front is greater than radius^2 are N, will be the maximum, so will take the first nsample points directly in the remaining points
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    # Considering that there may be points in the previous nsample points that are assigned N (ie, less than nsample points in the spherical area), this point needs to be discarded, and the first point can be used instead.
    # group_first: [B, S, k], actually copy the value of the first point in group_idx to the dimension of [B, S, K], which is convenient for subsequent replacement.
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    # Find the point where group_idx is equal to N
    mask = group_idx == N
    # Replace the value of these points with the value of the first point
    group_idx[mask] = group_first[mask]
    print('group idx shape',group_idx.shape)
    return group_idx

def extract_knn_patch(queries, pc, k):
    """
    queries [M, C]
    pc [P, C]
    """
    knn_search = NearestNeighbors(n_neighbors=k, algorithm='auto')
    knn_search.fit(pc)
    knn_idx = knn_search.kneighbors(queries, return_distance=False)
    k_patches = np.take(pc, knn_idx, axis=0)  # M, K, C
    return k_patches

def get_pairwise_distance(batch_features):
    # link: https://github.com/liruihui/PU-GAN/blob/master/Common/pc_util.py
    """Compute pairwise distance of a point cloud.
    Args:
      batch_features: numpy (batch_size, num_points, num_dims)
    Returns:
      pairwise distance: (batch_size, num_points, num_points)
    """

    og_batch_size = len(batch_features.shape)

    if og_batch_size == 2: #just two dimension
        batch_features = np.expand_dims(batch_features, axis=0)


    batch_features_transpose = np.transpose(batch_features, (0, 2, 1))

    #batch_features_inner = batch_features@batch_features_transpose
    batch_features_inner = np.matmul(batch_features,batch_features_transpose)

    #print(np.max(batch_features_inner), np.min(batch_features_inner))


    batch_features_inner = -2 * batch_features_inner
    batch_features_square = np.sum(np.square(batch_features), axis=-1, keepdims=True)


    batch_features_square_tranpose = np.transpose(batch_features_square, (0, 2, 1))

    return batch_features_square + batch_features_inner + batch_features_square_tranpose

def py_uniform_loss(points,idx,pts_cn,radius):
    '''
    from PU GAN: py version, link: https://github.com/liruihui/PU-GAN/blob/master/Common/loss_utils.py
    points: a batch of pcd
    idx??
    pts ??
    radius 1
    ----------------
    nsample : n_hat
    how to get n_hat
    pts_cn
    idx : should be returned value. which (B,M,nsample???), which is variable

    '''
    #print(type(idx))
    B,N,C = points.shape
    _,npoint,nsample = idx.shape
    uniform_vals = []
    for i in range(B):
        point = points[i]
        for j in range(npoint):
            number = pts_cn[i,j]
            coverage = np.square(number - nsample) / nsample
            if number<5:
                uniform_vals.append(coverage)
                continue
            _idx = idx[i, j, :number]
            disk_point = point[_idx]
            if disk_point.shape[0]<0:
                pair_dis = get_pairwise_distance(disk_point)#(batch_size, num_points, num_points)
                nan_valid = np.where(pair_dis<1e-7)
                pair_dis[nan_valid]=0
                pair_dis = np.squeeze(pair_dis, axis=0)
                pair_dis = np.sort(pair_dis, axis=1)
                shortest_dis = np.sqrt(pair_dis[:, 1])
            else:
                shortest_dis = pc_util.get_knn_dis(disk_point,disk_point,2)
                shortest_dis = shortest_dis[:,1]
            disk_area = math.pi * (radius ** 2) / disk_point.shape[0]
            #expect_d = math.sqrt(disk_area)
            expect_d = np.sqrt(2 * disk_area / 1.732)  # using hexagon
            dis = np.square(shortest_dis - expect_d) / expect_d
            uniform_val = coverage * np.mean(dis)

            uniform_vals.append(uniform_val)

    uniform_dis = np.array(uniform_vals).astype(np.float32)

    uniform_dis = np.mean(uniform_dis)
    return uniform_dis

### uniform loss from PU-GAN: link https://github.com/liruihui/PU-GAN/blob/master/Common/loss_utils.py
# def get_uniform_loss(pcd, percentages=[0.004,0.006,0.008,0.010,0.012], radius=1.0):
#     B, N, C = pcd.shape
#     npoint = int(N * 0.05)
#     loss=[]
#     for p in percentages:
#         # TODO to figure out the below equations ???
#         nsample = int(N*p)
#         r = math.sqrt(p*radius) # ??? not sure if it is corect or not
#         disk_area = math.pi *(radius ** 2) * p/nsample
#         #print(npoint,nsample)
#         new_xyz = gather_point(pcd, farthest_point_sample(npoint, pcd))  # (batch_size, npoint, 3)
#         idx, pts_cnt = query_ball_point(r, nsample, pcd, new_xyz)#(batch_size, npoint, nsample)