import cv2, os
import numpy as np
from tqdm import tqdm
import uuid
import imageio
import concurrent.futures

from src.utils.videoio import save_video_with_watermark 
from src.utils.videoio import load_video_to_cv2

def paste_pic(images, pic_path, crop_info, extended_crop=False, max_threads=8):

    if not os.path.isfile(pic_path):
        raise ValueError('pic_path must be a valid path to video/image file')
    elif pic_path.split('.')[-1] in ['jpg', 'png', 'jpeg']:
        # loader for first frame
        full_img = cv2.imread(pic_path)
    else:
        # loader for videos
        video_stream = cv2.VideoCapture(pic_path)
        fps = video_stream.get(cv2.CAP_PROP_FPS)
        full_frames = [] 
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break 
            break 
        full_img = frame
    frame_h = full_img.shape[0]
    frame_w = full_img.shape[1]

    crop_frames = images
    if type(images) == str and os.path.isfile(images):
        crop_frames = load_video_to_cv2(images)
    '''
    video_stream = cv2.VideoCapture(images)
    fps = video_stream.get(cv2.CAP_PROP_FPS)
    crop_frames = []
    while 1:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        crop_frames.append(frame)
    '''
    
    if len(crop_info) != 3:
        print("you didn't crop the image")
        return
    else:
        r_w, r_h = crop_info[0]
        clx, cly, crx, cry = crop_info[1]
        lx, ly, rx, ry = crop_info[2]
        lx, ly, rx, ry = int(lx), int(ly), int(rx), int(ry)
        # oy1, oy2, ox1, ox2 = cly+ly, cly+ry, clx+lx, clx+rx
        # oy1, oy2, ox1, ox2 = cly+ly, cly+ry, clx+lx, clx+rx

        if extended_crop:
            oy1, oy2, ox1, ox2 = cly, cry, clx, crx
        else:
            oy1, oy2, ox1, ox2 = cly+ly, cly+ry, clx+lx, clx+rx

    tmp_path = str(uuid.uuid4())+'.mp4'
    def process_image(crop_frame):
        p = cv2.resize(crop_frame.astype(np.uint8), (ox2-ox1, oy2 - oy1)) 

        mask = 255*np.ones(p.shape, p.dtype)
        location = ((ox1+ox2) // 2, (oy1+oy2) // 2)
        gen_img = cv2.seamlessClone(p, full_img, mask, location, cv2.NORMAL_CLONE)
        gen_img = cv2.cvtColor(gen_img, cv2.COLOR_BGR2RGB)
        return gen_img
    result = []
    #out_tmp = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_w, frame_h))
    with concurrent.futures.ThreadPoolExecutor(max_threads) as executor:
        for gen_img in tqdm(executor.map(process_image, crop_frames), total=len(crop_frames), desc='seamlessClone:'):
            result.append(gen_img)
    return result
    '''
    imageio.mimsave(tmp_path, result, fps=fps)
    #for img in result:
    #    out_tmp.write(img)
    #out_tmp.release()

    save_video_with_watermark(tmp_path, new_audio_path, full_video_path, watermark=False)
    os.remove(tmp_path)
    '''
