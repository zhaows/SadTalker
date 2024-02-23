import os
import sys
import hashlib
from flask import Flask, request
import json
import time
from argparse import ArgumentParser
import requests
import torch

app = Flask(__name__)
workPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
temp_path = os.path.join(workPath, 'temp')
util_path = os.path.join(workPath, 'utils')
tts_path = os.path.join(temp_path, 'tts')
asr_path = os.path.join(temp_path, 'asr')
img_path = os.path.join(temp_path, 'img')
metahuman_path = os.path.join(temp_path, 'metahuman')
sadTalker_path = '/algorithm/zhaoweisong/SadTalker'
print(temp_path, util_path, sadTalker_path)
sys.path.append(util_path)
sys.path.append(sadTalker_path)
sys.path.append(workPath)
from audio_base64 import base64_to_audio, audio2base64
# from audio_format import trans_audio_format
from inference_func import inference

def get_host_ip():
    import socket
    ip = None
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip

ip = get_host_ip()
if ip is None:
    print('get host ip failed, exit')
    sys.exit(-1)

root_url = 'http://172.29.0.162:5503'
def tts_paddlespeech(text, spk_id, timeout=100):
    request_id = '%s_%d'%(hashlib.md5(text.encode('utf-8')).hexdigest(), spk_id)
    tts_url = '%s/tts'%root_url
    print(tts_url)
    data = {'text': text,
            'request_id': request_id,
            'spk_id': spk_id}
    data = json.dumps(data)
    res = requests.post(tts_url, data, timeout=timeout)
    print(res.text)
    return res.text

def tts_microsoft(text, dialect='普通话', gender='Female', timeout=100):
    request_id = '%s_microsoft_%s'%(hashlib.md5(text.encode('utf-8')).hexdigest(), gender)
    tts_url = '%s/tts'%root_url
    print(tts_url)
    data = {'text': text,
            'source': 'microsoft',
            'gender': gender,
            'dialect': dialect,
            'request_id': request_id,
            'spk_id': 6}
    data = json.dumps(data)
    res = requests.post(tts_url, data, timeout=timeout)
    print(res.text)
    return res.text

# 视频生成默认参数设置
parser = ArgumentParser()
args = parser.parse_args()
args.driven_audio = ''
args.source_image = ''
args.result_dir = metahuman_path
args.enhancer = 'gfpgan'
args.enhancer = 'RestoreFormer'
args.preprocess = 'full'
args.still = True
args.background_enhancer = None
args.ref_eyeblink = None
args.ref_pose = None
args.checkpoint_dir = os.path.join(sadTalker_path, 'checkpoints')
args.pose_style = 0
args.batch_size = 8
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
if torch.cuda.is_available() and not args.cpu:
    args.device = "cuda"
else:
    args.device = "cpu"

@app.route('/metahuman', methods=['POST'])
def metahuman_func():
    start = time.time()
    # 解析request
    data = json.loads(request.data)
    # print('request body: %s'%data)

    # 判断audio参数
    if 'audio' in data:
        audio = data['audio']
        audio_path = os.path.join(asr_path, 'audio.wav')
        audio = base64_to_audio(audio, audio_path)
    else:
        # tts生成
        text = data['text']
        spk_id = data.get('spk_id', 6)
        gender = data.get('gender', 'Female')
        dialect = data.get('dialect', '普通话')
        tts_res = tts_microsoft(text, dialect=dialect, gender=gender)
        tts_res = json.loads(tts_res)
        audio_path = tts_res['result']['path']
        audio_url = tts_res['result']['url']
        print('text->%s, spk_id->%d, dialect->%s, gender->%s, tts_path->%s, audio_url->%s'%(text, spk_id, dialect, gender, audio_path, audio_url))
    print('audio_path: %s'%audio_path)

    # 获取img
    img = data['img']
    imgname = data['imgname']
    image_path = os.path.join(img_path, imgname)
    img = base64_to_audio(img, image_path)
    print('image_path: %s'%image_path)

    # 补充args
    args.driven_audio = audio_path
    args.source_image = image_path

    # 生成视频
    print(args)
    video_path = inference(args)
    video_name = os.path.basename(video_path)

    # response 
    video_url = 'http://%s:8000/metahuman/%s'%(ip, video_name)
    res = {'success': True,
           'code': 0,
           'message': {'global': 'success'},
           'result': {
               'video_url': video_url,
               'audio_url': '',
               'video': None}}
    print('server time cost: %f'%(time.time() - start))
    return res

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5507)
