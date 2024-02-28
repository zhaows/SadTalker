from glob import glob
import shutil
import torch
from time import  strftime
import os, sys, time
from argparse import ArgumentParser

from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff  
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path

workPath = os.path.dirname(os.path.abspath(__file__))
srcPath = os.path.join(workPath, 'src')

class SadTalker(object):
    def __init__(self, args):
        self.args = args
        self.checkpoint_dir = args.checkpoint_dir
        device = args.device
        sadtalker_paths = init_path(self.checkpoint_dir, os.path.join(srcPath, 'config'), args.size, args.old_version, args.preprocess)
        print(sadtalker_paths)
        #init model
        self.preprocess_model = CropAndExtract(sadtalker_paths, device)
        print('Preprocess model loaded')
        self.audio_to_coeff = Audio2Coeff(sadtalker_paths,  device)
        print('Audio2Coeff model loaded')
        self.animate_from_coeff = AnimateFromCoeff(sadtalker_paths, device, enhancer=args.enhancer, background_enhancer=args.background_enhancer)
        print('AnimateFromCoeff model loaded')

    def inference(self, args):
        #torch.backends.cudnn.enabled = False
        pic_path = args.source_image
        audio_path = args.driven_audio
        save_dir = os.path.join(args.result_dir, strftime("%Y_%m_%d_%H.%M.%S"))
        os.makedirs(save_dir, exist_ok=True)
        pose_style = args.pose_style
        device = args.device
        batch_size = args.batch_size
        input_yaw_list = args.input_yaw
        input_pitch_list = args.input_pitch
        input_roll_list = args.input_roll
        ref_eyeblink = args.ref_eyeblink
        ref_pose = args.ref_pose

        #crop image and extract 3dmm from image
        first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
        os.makedirs(first_frame_dir, exist_ok=True)
        print('3DMM Extraction for source image')
        first_coeff_path, crop_pic_path, crop_info =  self.preprocess_model.generate(pic_path, first_frame_dir, args.preprocess, source_image_flag=True, pic_size=args.size)
        if first_coeff_path is None:
            print("Can't get the coeffs of the input")
            return

        if ref_eyeblink is not None:
            ref_eyeblink_videoname = os.path.splitext(os.path.split(ref_eyeblink)[-1])[0]
            ref_eyeblink_frame_dir = os.path.join(save_dir, ref_eyeblink_videoname)
            os.makedirs(ref_eyeblink_frame_dir, exist_ok=True)
            print('3DMM Extraction for the reference video providing eye blinking')
            ref_eyeblink_coeff_path, _, _ =  self.preprocess_model.generate(ref_eyeblink, ref_eyeblink_frame_dir, args.preprocess, source_image_flag=False)
        else:
            ref_eyeblink_coeff_path=None

        if ref_pose is not None:
            if ref_pose == ref_eyeblink:
                ref_pose_coeff_path = ref_eyeblink_coeff_path
            else:
                ref_pose_videoname = os.path.splitext(os.path.split(ref_pose)[-1])[0]
                ref_pose_frame_dir = os.path.join(save_dir, ref_pose_videoname)
                os.makedirs(ref_pose_frame_dir, exist_ok=True)
                print('3DMM Extraction for the reference video providing pose')
                ref_pose_coeff_path, _, _ =  self.preprocess_model.generate(ref_pose, ref_pose_frame_dir, args.preprocess, source_image_flag=False)
        else:
            ref_pose_coeff_path=None

        #audio2ceoff
        batch = get_data(first_coeff_path, audio_path, device, ref_eyeblink_coeff_path, still=args.still)
        coeff_path = self.audio_to_coeff.generate(batch, save_dir, pose_style, ref_pose_coeff_path)

        # 3dface render
        if args.face3dvis:
            from src.face3d.visualize import gen_composed_video
            gen_composed_video(args, device, first_coeff_path, coeff_path, audio_path, os.path.join(save_dir, '3dface.mp4'))

        #coeff2video
        data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, audio_path,
                                    batch_size, input_yaw_list, input_pitch_list, input_roll_list,
                                    expression_scale=args.expression_scale, still_mode=args.still, preprocess=args.preprocess, size=args.size)

        result = self.animate_from_coeff.generate(data, save_dir, pic_path, crop_info, \
                                    enhancer=args.enhancer, background_enhancer=args.background_enhancer, preprocess=args.preprocess, img_size=args.size, batch_size=batch_size)

        shutil.move(result, save_dir+'.mp4')
        print('The generated video is named:', save_dir+'.mp4')

        if not args.verbose:
            shutil.rmtree(save_dir)
        return save_dir+'.mp4'
