import cv2
import os
import torch
from tqdm import tqdm
from basicsr.utils import img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from src.facerender.face_restoration_helper import FaceRestoreHelper
from torchvision.transforms.functional import normalize

from gfpgan.archs.gfpgan_bilinear_arch import GFPGANBilinear
from gfpgan.archs.gfpganv1_arch import GFPGANv1
from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean
import concurrent.futures

from facexlib.detection import init_detection_model
from facexlib.parsing import init_parsing_model
import threading
lock = threading.Lock()


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class GFPGANer():
    """Helper for restoration with GFPGAN.

    It will detect and crop faces, and then resize the faces to 512x512.
    GFPGAN is used to restored the resized faces.
    The background is upsampled with the bg_upsampler.
    Finally, the faces will be pasted back to the upsample background image.

    Args:
        model_path (str): The path to the GFPGAN model. It can be urls (will first download it automatically).
        upscale (float): The upscale of the final output. Default: 2.
        arch (str): The GFPGAN architecture. Option: clean | original. Default: clean.
        channel_multiplier (int): Channel multiplier for large networks of StyleGAN2. Default: 2.
        bg_upsampler (nn.Module): The upsampler for the background. Default: None.
    """

    def __init__(self, model_path, upscale=2, arch='clean', channel_multiplier=2, bg_upsampler=None, device=None):
        self.upscale = upscale
        self.bg_upsampler = bg_upsampler

        # initialize model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        # initialize the GFP-GAN
        if arch == 'clean':
            self.gfpgan = GFPGANv1Clean(
                out_size=512,
                num_style_feat=512,
                channel_multiplier=channel_multiplier,
                decoder_load_path=None,
                fix_decoder=False,
                num_mlp=8,
                input_is_latent=True,
                different_w=True,
                narrow=1,
                sft_half=True)
        elif arch == 'bilinear':
            self.gfpgan = GFPGANBilinear(
                out_size=512,
                num_style_feat=512,
                channel_multiplier=channel_multiplier,
                decoder_load_path=None,
                fix_decoder=False,
                num_mlp=8,
                input_is_latent=True,
                different_w=True,
                narrow=1,
                sft_half=True)
        elif arch == 'original':
            self.gfpgan = GFPGANv1(
                out_size=512,
                num_style_feat=512,
                channel_multiplier=channel_multiplier,
                decoder_load_path=None,
                fix_decoder=True,
                num_mlp=8,
                input_is_latent=True,
                different_w=True,
                narrow=1,
                sft_half=True)
        elif arch == 'RestoreFormer':
            from gfpgan.archs.restoreformer_arch import RestoreFormer
            self.gfpgan = RestoreFormer()
        # initialize face helper
        self.model_rootpath = 'gfpgan/weights'
        self.det_model = 'retinaface_resnet50'
        self.face_det = init_detection_model(self.det_model, half=False, device=self.device, model_rootpath=self.model_rootpath)
        self.face_parse = init_parsing_model(model_name='parsenet', device=self.device, model_rootpath=self.model_rootpath)


        if model_path.startswith('https://'):
            model_path = load_file_from_url(
                url=model_path, model_dir=os.path.join(ROOT_DIR, 'gfpgan/weights'), progress=True, file_name=None)
        loadnet = torch.load(model_path)
        if 'params_ema' in loadnet:
            keyname = 'params_ema'
        else:
            keyname = 'params'
        self.gfpgan.load_state_dict(loadnet[keyname], strict=True)
        self.gfpgan.eval()
        self.gfpgan = self.gfpgan.to(self.device)

    @torch.no_grad()
    def enhance_face(self, faces, batch_size = 8, weight=0.5):
        origin_h, origin_w = faces[0].shape[:2]
        def process_face(face):
            face = cv2.resize(face, (512, 512))
            face_t = img2tensor(face / 255., bgr2rgb=True, float32=True)
            normalize(face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            face_t = face_t.unsqueeze(0).to(self.device)
            return face_t

        max_threads = min(batch_size, len(faces))
        result = []
        with concurrent.futures.ThreadPoolExecutor(max_threads) as executor:
            for gen_img in tqdm(executor.map(process_face, faces), total=len(faces), desc='face enhance preprocess:'):
                result.append(gen_img)
        # face restoration
        restored_faces = []
        batchs = [result[i:i + batch_size] for i in range(0, len(result), batch_size)]
        for batch in tqdm(batchs, desc='face enhance:'):
            batch = torch.cat(batch, dim=0)
            output = self.gfpgan(batch, return_rgb=False, weight=weight)[0]
            for i in range(len(output)):
                restored_face = tensor2img(output[i].squeeze(0), rgb2bgr=True, min_max=(-1, 1))
                restored_face = restored_face.astype('uint8')
                restored_face = cv2.resize(restored_face, (origin_w, origin_h))
                restored_faces.append(restored_face)
        return restored_faces

    @torch.no_grad()
    def enhance(self, img, has_aligned=False, only_center_face=False, paste_back=True, weight=0.5):
        face_helper = FaceRestoreHelper(
            self.upscale,
            face_size=512,
            crop_ratio=(1, 1),
            det_model=self.det_model,
            save_ext='png',
            use_parse=True,
            device=self.device,
            model_rootpath=self.model_rootpath,
            face_det=self.face_det,
            face_parse=self.face_parse)

        face_helper.clean_all()

        if has_aligned:  # the inputs are already aligned
            img = cv2.resize(img, (512, 512))
            face_helper.cropped_faces = [img]
        else:
            face_helper.read_image(img)
            # get face landmarks for each face
            face_helper.get_face_landmarks_5(only_center_face=only_center_face, eye_dist_threshold=5)
            # eye_dist_threshold=5: skip faces whose eye distance is smaller than 5 pixels
            # TODO: even with eye_dist_threshold, it will still introduce wrong detections and restorations.
            # align and warp each face
            face_helper.align_warp_face()

        # face restoration
        for cropped_face in face_helper.cropped_faces:
            # prepare data
            cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(self.device)

            try:
                with lock:
                    output = self.gfpgan(cropped_face_t, return_rgb=False, weight=weight)[0]
                # convert to image
                restored_face = tensor2img(output.squeeze(0), rgb2bgr=True, min_max=(-1, 1))
            except RuntimeError as error:
                print(f'\tFailed inference for GFPGAN: {error}.')
                restored_face = cropped_face

            restored_face = restored_face.astype('uint8')
            face_helper.add_restored_face(restored_face)

        if not has_aligned and paste_back:
            # upsample the background
            if self.bg_upsampler is not None:
                # Now only support RealESRGAN for upsampling background
                bg_img = self.bg_upsampler.enhance(img, outscale=self.upscale)[0]
            else:
                bg_img = None

            face_helper.get_inverse_affine(None)
            # paste each restored face to the input image
            restored_img = face_helper.paste_faces_to_input_image(upsample_img=bg_img)
            #return face_helper.cropped_faces, face_helper.restored_faces, restored_img
        else:
            restored_img = None
            #return face_helper.cropped_faces, face_helper.restored_faces, None
        face_helper.clean_all()
        face_helper = None
        return None, None, restored_img