import os
import sys
import json
import time
from argparse import ArgumentParser

workPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
metahuman_path = os.path.join(workPath, 'metahuman')
print(workPath)
sadTalker_path = workPath
sys.path.append(sadTalker_path)
from inference_func import SadTalker 

class ObjDict(dict):
    """
    Makes a  dictionary behave like an object,with attribute-style access.
    """
    def __getattr__(self,name):
        try:
            return self[name]
        except:
            raise AttributeError(name)
    def __setattr__(self,name,value):
        self[name]=value


def get_args():
    # 视频生成默认参数设置
    args = ObjDict()
    args.driven_audio = ''
    args.source_image = ''
    args.result_dir = metahuman_path
    args.enhancer = 'gfpgan'
    args.enhancer = None
    # args.enhancer = 'RestoreFormer'
    args.preprocess = 'full'
    args.still = True
    args.background_enhancer = None
    args.ref_eyeblink = None
    args.ref_pose = None
    args.checkpoint_dir = os.path.join(sadTalker_path, 'checkpoints')
    args.pose_style = 8
    args.batch_size = 16
    args.size = 256
    args.expression_scale = 1.0
    args.input_yaw = None
    args.input_pitch = None
    args.input_roll = None
    args.cpu = False
    args.face3dvis = False
    args.verbose = False
    args.old_version = False
    args.net_recon = 'resnet50'
    args.init_path = None
    args.use_last_fc = False
    args.bfm_folder = os.path.join(sadTalker_path, 'checkpoints/BFM_Fitting')
    args.bfm_model = 'BFM_model_front.mat'
    args.focal = 1015.0
    args.center = 112.0
    args.camera_d = 10.0
    args.z_near = 5.0
    args.z_far = 15.0
    import torch
    if torch.cuda.is_available() and not args.cpu:
        args.device = "cuda"
    else:
        args.device = "cpu"
    return args

def main():
    audio_path = sys.argv[1]
    image_path = sys.argv[2]
    args = get_args()
    # 补充args
    args.driven_audio = audio_path
    args.source_image = image_path
    # 生成视频
    print(args)
    sadTalker = SadTalker(args)
    sadTalker.inference(args)

if __name__ == '__main__':
    main()
