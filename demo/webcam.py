import sys
from datetime import datetime
import time
import argparse
import cv2
from lib.preprocess import h36m_coco_format, revise_kpts
from lib.hrnet.gen_kpts import gen_video_kpts as hrnet_pose
import os 
import numpy as np
import torch
import glob
from tqdm import tqdm
import copy
from IPython import embed

sys.path.append(os.getcwd())
from model.strided_transformer import Model
from common.camera import *

import matplotlib
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from common.device import DEVICE

plt.switch_backend('agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def show2Dpose(kps, img):
    connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
                   [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
                   [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]

    LR = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)

    lcolor = (255, 0, 0)
    rcolor = (0, 0, 255)
    thickness = 5

    for j,c in enumerate(connections):
        start = map(int, kps[c[0]])
        end = map(int, kps[c[1]])
        start = list(start)
        end = list(end)
        cv2.line(img, (start[0], start[1]), (end[0], end[1]), lcolor if LR[j] else rcolor, thickness)
        cv2.circle(img, (start[0], start[1]), thickness=-1, color=(0, 255, 0), radius=5)
        cv2.circle(img, (end[0], end[1]), thickness=-1, color=(0, 255, 0), radius=5)

    return img


def show3Dpose(vals, ax):
    ax.view_init(elev=15., azim=70)

    I = np.array( [0, 0, 1, 4, 2, 5, 0, 7,  8,  8, 14, 15, 11, 12, 8,  9])
    J = np.array( [1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10])

    LR = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0,   1,  0,  0,  1,  1, 0, 0], dtype=bool)

    for i in np.arange( len(I) ):
        x, y, z = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)]

        ax.plot(x, y, z, lw=2)
        ax.scatter(x, y, z)

    RADIUS = 0.72
    RADIUS_Z = 0.7

    xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
    ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
    ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])
    ax.set_zlim3d([-RADIUS_Z+zroot, RADIUS_Z+zroot])
    ax.set_aspect('equal') # works fine in matplotlib==2.2.2 or 3.7.1

    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white) 
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)

    ax.tick_params('x', labelbottom = False)
    ax.tick_params('y', labelleft = False)
    ax.tick_params('z', labelleft = False)


def get_pose2D(video_source):
    print('\nGenerating 2D pose...')
    keypoints, scores = hrnet_pose(video_source, det_dim=416, num_peroson=1, gen_output=True)
    keypoints, scores, valid_frames = h36m_coco_format(keypoints, scores)
    return keypoints


def img2video(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) + 5

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    names = sorted(glob.glob(os.path.join(output_dir + 'pose/', '*.png')))
    img = cv2.imread(names[0])
    size = (img.shape[1], img.shape[0])

    videoWrite = cv2.VideoWriter(output_dir + video_name + '.mp4', fourcc, fps, size) 

    for name in names:
        img = cv2.imread(name)
        videoWrite.write(img)

    videoWrite.release()


def showimage(ax, img):
    ax.set_xticks([])
    ax.set_yticks([]) 
    plt.axis('off')
    ax.imshow(img)


def get_pose3D(img, output_dir, keypoints):
    args, _ = argparse.ArgumentParser().parse_known_args()
    args.layers, args.channel, args.d_hid, args.frames = 3, 256, 512, 351
    args.pad = (args.frames - 1) // 2
    args.stride_num = [3, 9, 13]
    args.previous_dir = 'checkpoint/pretrained'
    args.n_joints, args.out_joints = 17, 17

    ## Reload
    model = Model(args).to(DEVICE)

    model_dict = model.state_dict()
    model_paths = sorted(glob.glob(os.path.join(args.previous_dir, '*.pth')))
    for path in model_paths:
        if os.path.split(path)[-1][0] == 'n':
            model_path = path

    pre_dict = torch.load(model_path, map_location=DEVICE)
    for name, key in model_dict.items():
        model_dict[name] = pre_dict[name]
    model.load_state_dict(model_dict)

    model.eval()

    ## 3D
    print('\nGenerating 3D pose...')
    img_size = img.shape

    ## input frames
    start = 0
    end = 0
    pad = args.pad
    left_pad = pad - start
    right_pad = pad - end

    input_2D_no = keypoints[0][start:end+1]
    input_2D_no = np.pad(input_2D_no, ((left_pad, right_pad), (0, 0), (0, 0)), 'edge')

    joints_left =  [4, 5, 6, 11, 12, 13]
    joints_right = [1, 2, 3, 14, 15, 16]

    input_2D = normalize_screen_coordinates(input_2D_no, w=img_size[1], h=img_size[0])  

    input_2D_aug = copy.deepcopy(input_2D)
    input_2D_aug[ :, :, 0] *= -1
    input_2D_aug[ :, joints_left + joints_right] = input_2D_aug[ :, joints_right + joints_left]
    input_2D = np.concatenate((np.expand_dims(input_2D, axis=0), np.expand_dims(input_2D_aug, axis=0)), 0)

    input_2D = input_2D[np.newaxis, :, :, :, :]

    input_2D = torch.from_numpy(input_2D.astype('float32')).to(DEVICE)

    N = input_2D.size(0)
    print(input_2D.shape)

    ## estimation
    output_3D_non_flip, _ = model(input_2D[:, 0])
    output_3D_flip, _     = model(input_2D[:, 1])

    output_3D_flip[:, :, :, 0] *= -1
    output_3D_flip[:, :, joints_left + joints_right, :] = output_3D_flip[:, :, joints_right + joints_left, :] 

    output_3D = (output_3D_non_flip + output_3D_flip) / 2

    output_3D[:, :, 0, :] = 0
    post_out = output_3D[0, 0].cpu().detach().numpy()

    rot =  [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
    rot = np.array(rot, dtype='float32')
    post_out = camera_to_world(post_out, R=rot, t=0)
    post_out[:, 2] -= np.min(post_out[:, 2])

    input_2D_no = input_2D_no[args.pad]

    ## 2D
    image = show2Dpose(input_2D_no, copy.deepcopy(img))

    print('Generating 3D pose successful!')

    ## all
    print('\nGenerating demo...')
    image_2d = image
    image_3d = post_out

    ## crop
    edge = (image_2d.shape[1] - image_2d.shape[0]) // 2
    image_2d = image_2d[:, edge:image_2d.shape[1] - edge]

    # edge = 102
    # print('image_3d.shape', image_3d.shape)
    # image_3d = image_3d[edge:image_3d.shape[0] - edge, edge:image_3d.shape[1] - edge]

    ## show
    font_size = 12
    fig = plt.figure(figsize=(9.6, 5.4))
    ax = plt.subplot(121)
    showimage(ax, image_2d)
    ax.set_title("Input", fontsize = font_size)

    ax = plt.subplot(122)
    showimage(ax, image_3d)
    ax.set_title("Reconstruction", fontsize = font_size)

    ## save
    output_dir_pose = output_dir +'pose/'
    os.makedirs(output_dir_pose, exist_ok=True)
    plt.savefig(output_dir_pose + str(datetime.now()) + '_pose.png', dpi=200, bbox_inches = 'tight')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    start = time.time()
    parser.add_argument('--gpu', type=str, default='0', help='input available gpus')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    video_source = 0    
    video_name = f'webcam-{video_source}'
    output_dir = './demo/output/' + video_name + '/'
    cap = cv2.VideoCapture(video_source)
    
    for i in range(10):
        ret, frame = cap.read()
        if not ret:
            break
        keypoints = get_pose2D(video_source)
        get_pose3D(frame, output_dir, keypoints)        
    img2video(video_source, output_dir)
    print('Generating demo successful!')
    duration = time.time() - start
    print(f'duration:{duration:.4f}')


