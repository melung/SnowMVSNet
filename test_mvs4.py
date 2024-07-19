import argparse, os, time, sys, gc, cv2
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from datasets import find_dataset_def
from models import *
from utils import *
from datasets.data_io import read_pfm, save_pfm
from plyfile import PlyData, PlyElement
from PIL import Image
import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import partial
import signal

cudnn.benchmark = True
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"

parser = argparse.ArgumentParser(description='Predict depth, filter, and fuse')
parser.add_argument('--model', default='mvsnet', help='select model')

parser.add_argument('--dataset', default='dtu_yao_eval', help='select dataset')
parser.add_argument('--testpath', help='testing data dir for some scenes')
parser.add_argument('--testlist', help='testing scene list')

parser.add_argument('--batch_size', type=int, default=5, help='testing batch size')

parser.add_argument('--loadckpt', default=None, help='load a specific checkpoint')
parser.add_argument('--outdir', default='./outputs', help='output dir')

parser.add_argument('--share_cr', action='store_true', help='whether share the cost volume regularization')

parser.add_argument('--ndepths', type=str, default="8,8,4,4", help='ndepths')
parser.add_argument('--depth_inter_r', type=str, default="0.5,0.5,0.5,1", help='depth_intervals_ratio')

parser.add_argument('--interval_scale', type=float, required=True, help='the depth interval scale')
parser.add_argument('--num_view', type=int, default=5, help='num of view')
parser.add_argument('--max_h', type=int, default=1200, help='testing max h')
parser.add_argument('--max_w', type=int, default=1600, help='testing max w')
parser.add_argument('--fix_res', action='store_true', help='scene all using same res')

parser.add_argument('--num_worker', type=int, default=4, help='depth_filer worker')
parser.add_argument('--save_freq', type=int, default=500, help='save freq of local pcd')

parser.add_argument('--filter_method', type=str, default='gipuma', choices=["gipuma", "normal"], help="filter method")

#filter
parser.add_argument('--conf', type=float, default=0.3, help='prob confidence')
parser.add_argument('--thres_view', type=int, default=4, help='threshold of num view')

parser.add_argument("--fpn_base_channel", type=int, default=8)
parser.add_argument("--reg_channel", type=int, default=8)
parser.add_argument('--reg_mode', type=str, default="reg2d")
parser.add_argument('--dlossw', type=str, default="1,1,1,1", help='depth loss weight for different stage')
parser.add_argument('--resume', action='store_true', help='continue to train the model')
parser.add_argument('--group_cor', action='store_true',help='group correlation')
parser.add_argument('--group_cor_dim', type=str, default="8,8,4,4", help='group correlation dim')
parser.add_argument('--inverse_depth', action='store_true',help='inverse depth')
parser.add_argument('--agg_type', type=str, default="ConvBnReLU3D", help='cost regularization type')
parser.add_argument('--dcn', action='store_true',help='dcn')
parser.add_argument('--arch_mode', type=str, default="fpn")
parser.add_argument('--ot_continous', action='store_true',help='optimal transport continous gt bin')
parser.add_argument('--ot_eps', type=float, default=1)
parser.add_argument('--ot_iter', type=int, default=0)
parser.add_argument('--rt', action='store_true',help='robust training')
parser.add_argument('--use_raw_train', action='store_true',help='using 1200x1600 training')
parser.add_argument('--mono', action='store_true',help='query to build mono depth prediction and loss')
parser.add_argument('--split', type=str, default='intermediate', help='intermediate or advanced')
parser.add_argument('--save_jpg', action='store_true')
parser.add_argument('--ASFF', action='store_true')
parser.add_argument('--vis_ETA', action='store_true')
parser.add_argument('--vis_mono', action='store_true')
parser.add_argument('--attn_temp', type=float, default=2)

# parse arguments and check
args = parser.parse_args()
print("argv:", sys.argv[1:])
print_args(args)

if args.use_raw_train:
    args.max_h = 1200
    args.max_w = 1600
    
num_stage = len([int(nd) for nd in args.ndepths.split(",") if nd])

Interval_Scale = args.interval_scale
print("***********Interval_Scale**********\n", Interval_Scale)


# read intrinsics and extrinsics
def read_camera_parameters(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
    return intrinsics, extrinsics


# read an image
def read_img(filename):
    img = Image.open(filename)
    # scale 0~255 to 0~1
    np_img = np.array(img, dtype=np.float32) / 255.
    return np_img


# read a binary mask
def read_mask(filename):
    return read_img(filename) > 0.5


# save a binary mask
def save_mask(filename, mask):
    #assert mask.dtype == np.bool
    mask = mask.astype(np.uint8) * 255
    Image.fromarray(mask).save(filename)


# read a pair file, [(ref_view1, [src_view1-1, ...]), (ref_view2, [src_view2-1, ...]), ...]
def read_pair_file(filename):
    data = []
    with open(filename) as f:
        num_viewpoint = int(f.readline())
        # 49 viewpoints
        for view_idx in range(num_viewpoint):
            ref_view = int(f.readline().rstrip())
            src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
            if len(src_views) > 0:
                data.append((ref_view, src_views))
    return data

def write_cam(file, cam):
    f = open(file, "w")
    f.write('extrinsic\n')
    for i in range(0, 4):
        for j in range(0, 4):
            f.write(str(cam[0][i][j]) + ' ')
        f.write('\n')
    f.write('\n')

    f.write('intrinsic\n')
    for i in range(0, 3):
        for j in range(0, 3):
            f.write(str(cam[1][i][j]) + ' ')
        f.write('\n')

    f.write('\n' + str(cam[1][3][0]) + ' ' + str(cam[1][3][1]) + ' ' + str(cam[1][3][2]) + ' ' + str(cam[1][3][3]) + '\n')

    f.close()

def save_depth(testlist):
    torch.cuda.reset_peak_memory_stats()
    total_time = 0
    total_sample = 0
    for scene in testlist:
        time_this_scene, sample_this_scene = save_scene_depth([scene])
        total_time += time_this_scene
        total_sample += sample_this_scene
    gpu_measure = torch.cuda.max_memory_allocated() / 1024. / 1024.
    print('avg time: {}'.format(total_time/total_sample))
    print('max gpu: {}'.format(gpu_measure))


def save_scene_depth(testlist):
    # dataset, dataloader
    MVSDataset = find_dataset_def(args.dataset)
    test_dataset = MVSDataset(args.testpath, testlist, "test", args.num_view, Interval_Scale,
                            max_h=args.max_h, max_w=args.max_w, fix_res=args.fix_res)
    TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=4, drop_last=False)

    # model
    model = MVS4net(arch_mode=args.arch_mode, reg_net=args.reg_mode, num_stage=4, 
                    fpn_base_channel=args.fpn_base_channel, reg_channel=args.reg_channel, 
                    stage_splits=[int(n) for n in args.ndepths.split(",")], 
                    depth_interals_ratio=[float(ir) for ir in args.depth_inter_r.split(",")],
                    group_cor=args.group_cor, group_cor_dim=[int(n) for n in args.group_cor_dim.split(",")],
                    inverse_depth=args.inverse_depth,
                    agg_type=args.agg_type,
                    dcn=args.dcn,
                    mono=args.mono,
                    asff=args.ASFF,
                    attn_temp=args.attn_temp,
                    vis_ETA=args.vis_ETA,
                    vis_mono=args.vis_mono
                )
    # load checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict['model'], strict=True)
    model = nn.DataParallel(model)
    model.cuda()
    model.eval()
    
    total_time = 0
    with torch.no_grad():
        for batch_idx, sample in enumerate(TestImgLoader):
            sample_cuda = tocuda(sample)
            start_time = time.time()
            outputs = model(sample_cuda["imgs"], sample_cuda["R"], sample_cuda["proj_matrices"], sample_cuda["depth_values"], sample["filename"])
            end_time = time.time()
            total_time += end_time - start_time
            outputs = tensor2numpy(outputs)
            del sample_cuda
            forward_max_memory_allocated = torch.cuda.max_memory_allocated() / (1024.0 ** 2)

            filenames = sample["filename"]
            cams = sample["proj_matrices"]["stage{}".format(num_stage)].numpy()
            imgs = sample["imgs"]
            imgs = np.transpose(imgs,(1,0,2,3,4))

            depth_values = sample["depth_values"].numpy()
            print('Iter {}/{}, Time:{} Res:{}'.format(batch_idx, len(TestImgLoader), end_time - start_time, imgs[0].shape))
            print("mem",forward_max_memory_allocated)
            # save depth maps and confidence maps
            # weight, vals, intra_w, hypo, att
            for filename, cam, img, depth_est, normal_est, photometric_confidence, weight, vals in zip(filenames, cams, imgs,  \
                                                            outputs["depth"], outputs["normal_plane"], outputs["photometric_confidence"],outputs["weight"],outputs["val"]):
                #imgs = img.numpy()
                imgs = img.copy()
                img = img[0]  #ref view
                cam = cam[0]  #ref cam
                depth_filename = os.path.join(args.outdir, filename.format('depth_est', '.pfm'))
                confidence_filename = os.path.join(args.outdir, filename.format('confidence', '.pfm'))
                normal_filename = os.path.join(args.outdir, filename.format('normal', '.pfm'))
                cam_filename = os.path.join(args.outdir, filename.format('cams', '_cam.txt'))
                img_filename = os.path.join(args.outdir, filename.format('images', '.jpg'))
                #ply_filename = os.path.join(args.outdir, filename.format('ply_local', '.ply'))
                #weight_filename = os.path.join(args.outdir, filename.format('weight', '.pfm'))
                #valid_vol_filename = os.path.join(args.outdir, filename.format('valid_volume', '.pfm'))

                
                # prob_volume_filename = os.path.join(args.outdir, filename.format('prob_vol', '.npy'))
                # hypo_volume_filename = os.path.join(args.outdir, filename.format('hypo_vol', '.npy'))
                
                
                os.makedirs(depth_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(normal_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(confidence_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(cam_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(img_filename.rsplit('/', 1)[0], exist_ok=True)
                #os.makedirs(ply_filename.rsplit('/', 1)[0], exist_ok=True)
                #os.makedirs(weight_filename.rsplit('/', 1)[0], exist_ok=True)
                #os.makedirs(valid_vol_filename.rsplit('/', 1)[0], exist_ok=True)

                # os.makedirs(prob_volume_filename.rsplit('/', 1)[0], exist_ok=True)
                # os.makedirs(hypo_volume_filename.rsplit('/', 1)[0], exist_ok=True)
                
                #save depth maps
                save_pfm(depth_filename, depth_est)
                plt.imsave(depth_filename[:-3]+".jpg", depth_est, cmap="plasma", vmin=depth_values[0,0], vmax=depth_values[0,-1])
                normal_est = normal_est.transpose((1,2,0))

                #save_pfm(normal_filename, normal_est)
                #normal_est[:,:,:] = 255*(normal_est[:,:,:] - np.min(normal_est[:,:,:]))/(np.max(normal_est[:,:,:])-np.min(normal_est[:,:,:]))
                normal_est[:,:,:] = 127.5*(normal_est[:,:,:] + 1.0)

                normal_est = cv2.cvtColor(normal_est,cv2.COLOR_RGB2BGR)
                cv2.imwrite(normal_filename[:-3]+"jpg", normal_est)
                
                #save confidence maps
                confidence_list = [outputs['stage{}'.format(i)]['photometric_confidence'].squeeze(0) for i in range(1,5)]
                val_vol_list = [outputs['stage{}'.format(i)]['valid_volume'].squeeze(0) for i in range(1,5)]

                photometric_confidence = confidence_list[-1]  # H W
                val = np.mean(val_vol_list[-1], axis = 0)/4.0
                w_val = val + 0.0
                w_val[val == 0] = 0.0
                photometric_confidence[val == 0] = 0.0
                save_pfm(confidence_filename, photometric_confidence)

                #save_pfm(valid_vol_filename, val) 
                #np.save(hypo_volume_filename, hypo)
                # print("hypo",np.shape(hypo))
                # #np.save(prob_volume_filename, att)
                # print("prob",np.shape(att))
                
                #save cams, img
                write_cam(cam_filename, cam)
                img = np.clip(np.transpose((0.5*img) + 0.5, (1, 2, 0)) * 255, 0, 255).astype(np.uint8)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(img_filename, img_bgr)

                # sw = np.sum(weight,axis = 0) + 1e-8


                # for i in range(weight.shape[0]):
                #     src_img = imgs[i+1]
                #     src_img = np.clip(np.transpose((0.5*src_img) + 0.5, (1, 2, 0)) * 255, 0, 255).astype(np.uint8)
                #     src_img = cv2.cvtColor(src_img, cv2.COLOR_RGB2BGR)
                    
                #     #weight_filename = os.path.join(args.outdir, filename.format('weight','view'+str(i)+'.pfm'))
                #     weight_img_filename = os.path.join(args.outdir, filename.format('weight','view'+str(i+1)+'.jpg'))
                #     weight_n_img_filename = os.path.join(args.outdir, filename.format('weight','_n_view'+str(i+1)+'.jpg'))
                #     src_img_filename = os.path.join(args.outdir, filename.format('weight','view'+str(i+1)+'_image.jpg'))
                #     #cv2.imwrite(src_img_filename, src_img)
                #     #save_pfm(weight_filename, weight[i,0,:,:])
                #     ma = np.mean(vals[i,0,:,:,:],axis=0)
                #     #plt.imsave(weight_img_filename, weight[i,0,:,:]*ma , cmap="gray", vmin=0, vmax=1)
                #     #plt.imsave(weight_n_img_filename, ma*weight[i,0,:,:]/sw[0,:,:], cmap="gray", vmin=0, vmax=0.5)



                # if batch_idx % args.save_freq == 0:
                #     generate_pointcloud(img, depth_est, ply_filename, cam[1, :3, :3])

            torch.cuda.empty_cache()
            gc.collect()
    return total_time, len(TestImgLoader)



# project the reference point cloud into the source view, then project back
def reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    ## step1. project reference pixels to the source view
    # reference view x, y
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])
    # reference 3D space
    xyz_ref = np.matmul(np.linalg.inv(intrinsics_ref),
                        np.vstack((x_ref, y_ref, np.ones_like(x_ref))) * depth_ref.reshape([-1]))
    # source 3D space
    xyz_src = np.matmul(np.matmul(extrinsics_src, np.linalg.inv(extrinsics_ref)),
                        np.vstack((xyz_ref, np.ones_like(x_ref))))[:3]
    # source view x, y
    K_xyz_src = np.matmul(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:2] / K_xyz_src[2:3]

    ## step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = xy_src[0].reshape([height, width]).astype(np.float32)
    y_src = xy_src[1].reshape([height, width]).astype(np.float32)
    sampled_depth_src = cv2.remap(depth_src, x_src, y_src, interpolation=cv2.INTER_LINEAR)
    # mask = sampled_depth_src > 0

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xyz_src = np.matmul(np.linalg.inv(intrinsics_src),
                        np.vstack((xy_src, np.ones_like(x_ref))) * sampled_depth_src.reshape([-1]))
    # reference 3D space
    xyz_reprojected = np.matmul(np.matmul(extrinsics_ref, np.linalg.inv(extrinsics_src)),
                                np.vstack((xyz_src, np.ones_like(x_ref))))[:3]
    # source view x, y, depth
    depth_reprojected = xyz_reprojected[2].reshape([height, width]).astype(np.float32)
    K_xyz_reprojected = np.matmul(intrinsics_ref, xyz_reprojected)
    xy_reprojected = K_xyz_reprojected[:2] / K_xyz_reprojected[2:3]
    x_reprojected = xy_reprojected[0].reshape([height, width]).astype(np.float32)
    y_reprojected = xy_reprojected[1].reshape([height, width]).astype(np.float32)

    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src


def check_geometric_consistency(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref,
                                                     depth_src, intrinsics_src, extrinsics_src)
    # check |p_reproj-p_1| < 1
    dist = np.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)

    # check |d_reproj-d_1| / d_1 < 0.01
    depth_diff = np.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref

    mask = np.logical_and(dist < 0.2, relative_depth_diff < 0.01)
    depth_reprojected[~mask] = 0

    return mask, depth_reprojected, x2d_src, y2d_src


def filter_depth(pair_folder, scan_folder, out_folder, plyfilename):
    # the pair file
    pair_file = os.path.join(pair_folder, "pair.txt")
    # for the final point cloud
    vertexs = []
    vertex_colors = []

    pair_data = read_pair_file(pair_file)

    # for each reference view and the corresponding source views
    for ref_view, src_views in pair_data:
        # src_views = src_views[:args.num_view]
        # load the camera parameters
        ref_intrinsics, ref_extrinsics = read_camera_parameters(
            os.path.join(scan_folder, 'cams/{:0>8}_cam.txt'.format(ref_view)))
        # load the reference image
        ref_img = read_img(os.path.join(scan_folder, 'images/{:0>8}.jpg'.format(ref_view)))
        # load the estimated depth of the reference view
        ref_depth_est = read_pfm(os.path.join(out_folder, 'depth_est/{:0>8}.pfm'.format(ref_view)))[0]
        # load the photometric mask of the reference view
        confidence = read_pfm(os.path.join(out_folder, 'confidence/{:0>8}.pfm'.format(ref_view)))[0]
        photo_mask = confidence > args.conf

        all_srcview_depth_ests = []
        all_srcview_x = []
        all_srcview_y = []
        all_srcview_geomask = []

        # compute the geometric mask
        geo_mask_sum = 0
        for src_view in src_views:
            # camera parameters of the source view
            src_intrinsics, src_extrinsics = read_camera_parameters(
                os.path.join(scan_folder, 'cams/{:0>8}_cam.txt'.format(src_view)))
            # the estimated depth of the source view
            src_depth_est = read_pfm(os.path.join(out_folder, 'depth_est/{:0>8}.pfm'.format(src_view)))[0]

            geo_mask, depth_reprojected, x2d_src, y2d_src = check_geometric_consistency(ref_depth_est, ref_intrinsics, ref_extrinsics,
                                                                      src_depth_est,
                                                                      src_intrinsics, src_extrinsics)
            geo_mask_sum += geo_mask.astype(np.int32)
            all_srcview_depth_ests.append(depth_reprojected)
            all_srcview_x.append(x2d_src)
            all_srcview_y.append(y2d_src)
            all_srcview_geomask.append(geo_mask)

        depth_est_averaged = (sum(all_srcview_depth_ests) + ref_depth_est) / (geo_mask_sum + 1)
        # at least 3 source views matched
        geo_mask = geo_mask_sum >= args.thres_view
        final_mask = np.logical_and(photo_mask, geo_mask)

        os.makedirs(os.path.join(out_folder, "mask"), exist_ok=True)
        save_mask(os.path.join(out_folder, "mask/{:0>8}_photo.png".format(ref_view)), photo_mask)
        save_mask(os.path.join(out_folder, "mask/{:0>8}_geo.png".format(ref_view)), geo_mask)
        save_mask(os.path.join(out_folder, "mask/{:0>8}_final.png".format(ref_view)), final_mask)

        print("processing {}, ref-view{:0>2}, photo/geo/final-mask:{}/{}/{}".format(scan_folder, ref_view,
                                                                                    photo_mask.mean(),
                                                                                    geo_mask.mean(), final_mask.mean()))

        height, width = depth_est_averaged.shape[:2]
        x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))
        # valid_points = np.logical_and(final_mask, ~used_mask[ref_view])
        valid_points = final_mask
        print("valid_points", valid_points.mean())
        x, y, depth = x[valid_points], y[valid_points], depth_est_averaged[valid_points]
        #color = ref_img[1:-16:4, 1::4, :][valid_points]  # hardcoded for DTU dataset
        color = ref_img[valid_points]

        xyz_ref = np.matmul(np.linalg.inv(ref_intrinsics),
                            np.vstack((x, y, np.ones_like(x))) * depth)
        xyz_world = np.matmul(np.linalg.inv(ref_extrinsics),
                              np.vstack((xyz_ref, np.ones_like(x))))[:3]
        vertexs.append(xyz_world.transpose((1, 0)))
        vertex_colors.append((color * 255).astype(np.uint8))


    vertexs = np.concatenate(vertexs, axis=0)
    vertex_colors = np.concatenate(vertex_colors, axis=0)
    vertexs = np.array([tuple(v) for v in vertexs], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex_colors = np.array([tuple(v) for v in vertex_colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)
    for prop in vertexs.dtype.names:
        vertex_all[prop] = vertexs[prop]
    for prop in vertex_colors.dtype.names:
        vertex_all[prop] = vertex_colors[prop]

    el = PlyElement.describe(vertex_all, 'vertex')
    PlyData([el]).write(plyfilename)
    print("saving the final model to", plyfilename)


def init_worker():
    '''
    Catch Ctrl+C signal to termiante workers
    '''
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def pcd_filter_worker(scan):

    save_name = '{}.ply'.format(scan)
    pair_folder = os.path.join(args.testpath, scan)
    scan_folder = os.path.join(args.outdir, scan)
    out_folder = os.path.join(args.outdir, scan)
    filter_depth(pair_folder, scan_folder, out_folder, os.path.join(args.outdir, save_name))


def pcd_filter(testlist, number_worker):

    partial_func = partial(pcd_filter_worker)

    p = Pool(number_worker, init_worker)
    try:
        p.map(partial_func, testlist)
    except KeyboardInterrupt:
        print("....\nCaught KeyboardInterrupt, terminating workers")
        p.terminate()
    else:
        p.close()
    p.join()

def mrun_rst(eval_dir, plyPath):
    print('Runing BaseEvalMain_func.m...')
    os.chdir(eval_dir)
    os.system('/mnt/cfs/algorithm/xiaofeng.wang/jeff/code/MVS/misc/matlab/bin/matlab -nodesktop -nosplash -r "BaseEvalMain_func(\'{}\'); quit" '.format(plyPath))
    print('Runing ComputeStat_func.m...')
    os.system('/mnt/cfs/algorithm/xiaofeng.wang/jeff/code/MVS/misc/matlab/bin/matlab -nodesktop -nosplash -r "ComputeStat_func(\'{}\'); quit" '.format(plyPath))
    print('Check your results! ^-^')

if __name__ == '__main__':

    if args.vis_ETA:
        os.makedirs('./debug_figs/vis_ETA', exist_ok=True)

    if args.testlist != "all":
        with open(args.testlist) as f:
            content = f.readlines()
            testlist = [line.rstrip() for line in content]

    # step1. save all the depth maps and the masks in outputs directory
    save_depth(testlist)

    # step2. fusion pointcloud use general method
    # if args.dataset.startswith('general'):
    #     # step2. filter saved depth maps with photometric confidence maps and geometric constraints
    #     pcd_filter(testlist, args.num_worker)

    #     # Make sure the matlab is installed and you can comment out the following lines
    #     # And you also need to change the path of the matlab script

    #     mrun_rst(
    #         eval_dir='./evaluations/dtu/',
    #         plyPath='./'+args.outdir[1:]
    #     )


 