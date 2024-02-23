import os
import sys
import hashlib
from flask import Flask, request
import json
import time
from tts_microsoft import tts

app = Flask(__name__)
temp_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'temp')
util_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'utils')
tts_path = os.path.join(temp_path, 'tts')
asr_path = os.path.join(temp_path, 'asr')
print(temp_path, util_path)
sys.path.append(util_path)
from audio_base64 import base64_to_audio
from audio_format import trans_audio_format

# paddlespeech
import paddle
from paddlespeech.cli.tts.infer_user import TTSExecutor
from paddlespeech.cli.asr.infer import ASRExecutor

tts_executor = TTSExecutor()
asr_executor = ASRExecutor()

import socket
def get_host_ip():
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

@app.route('/asr', methods=['POST'])
def asr_func():
    start = time.time()
    # 解析request
    data = json.loads(request.data)
    #print('request body: %s'%data)

    request_id = data.get('request_id', '')
    audio = data['audio']
    audio_type = data['audio_type']
    output_path = os.path.join(asr_path,
                               '%s.%s'%(hashlib.md5(audio.encode('utf-8')).hexdigest(),
                                        audio_type))
    audio = base64_to_audio(audio, output_path)
    ## trans audio format
    if audio_type != 'wav':
        output_path = trans_audio_format(output_path, audio_type, 'wav')
    
    print(output_path, audio_type)
    # asr
    result = asr_executor(
            audio_file=output_path,
            force_yes = True)
    res = {'success': True,
           'code': 0,
           'message': {'global': 'success'},
           'result': {
               'text': result}}
    print('server time cost: %f'%(time.time() - start))
    return res

@app.route('/tts', methods=['POST'])
def tts_func():
    start = time.time()
    # 解析request
    data = json.loads(request.data)
    print('request body: %s'%data)

    source = data['source']
    request_id = data.get('request_id', 'output')
    audio_name = '%s.wav'%request_id
    outputPath = os.path.join(tts_path, audio_name)
    text = data['text']
    lang = data.get('lang', 'zh')
    dialect = data.get('dialect', '普通话')
    if source == 'microsoft':
        if lang == 'zh':
            lang = 'zh-CN'
        gender = data.get('gender', 'Female')
        if gender == 'Female':
            if dialect == '普通话':
                name = 'zh-CN-XiaoxiaoNeural'
                style = 'newscast'
            elif dialect == '吴语':
                name = 'wuu-CN-XiaotongNeural'
                style = ''
            elif dialect == '西南官话':
                name = ''
                style = ''
            elif dialect == '粤语':
                name = 'yue-CN-XiaoMinNeural'
                style = ''
            else:
                pass
        else:
            if dialect == '普通话':
                name = 'zh-CN-YunxiNeural'
                style = 'newscast'
            elif dialect == '吴语':
                name = 'wuu-CN-YunzheNeural'
                style = ''
            elif dialect == '西南官话':
                name = 'zh-CN-sichuan-YunxiNeural'
                style = ''
            elif dialect == '粤语':
                name = 'yue-CN-YunSongNeural'
                style = ''
            else:
                pass

        tts(text, outputPath, lang, name, style)
    else:
        am = 'fastspeech2_aishell3'
        voc = 'pwgan_aishell3'
        spk_id = data.get('spk_id', '6')
        if lang == 'mix':
            am = 'fastspeech2_mix'
            voc = 'pwgan_aishell3'

        print('params: am->%s, voc->%s, spk_id->%s, lang->%s, audio_name->%s'%(am, voc, spk_id, lang, audio_name))
        # tts合成
        wav_file = tts_executor(
                text = text,
                output = outputPath,
                am = am,
                voc = voc,
                spk_id = spk_id,
                lang = lang,
                use_onnx = True,
                cpu_threads=16)
    # response 
    audio_path = 'http://%s:8000/tts/%s'%(ip, audio_name)
    res = {'success': True,
           'code': 0,
           'message': {'global': 'success'},
           'result': {
               'lang': lang,
               'url': audio_path,
               'path': outputPath,
               'audio': None}}
    print('server time cost: %f'%(time.time() - start))
    return res

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5503)
