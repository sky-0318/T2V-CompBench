import argparse
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

import json
import cv2
from torchvision.io import write_video

def extract_frames(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= num_frames:
        frame_indices = np.arange(total_frames)
    else:
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    for i in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def rgb_to_yuv(frame):
    yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return yuv_frame


def frames_to_video(frames, output_path, fps=8):
    yuv_frames = [rgb_to_yuv(frame) for frame in frames]
    video_tensor = torch.from_numpy(np.array(yuv_frames)).to(torch.uint8)
    write_video(output_path, video_tensor, fps, video_codec='h264', options={'crf': '18'})


def convert_video(input_path, output_path):
    frames = extract_frames(input_path)
    frames_to_video(frames, output_path)


def video2img(video_path,frames_dir,question):
    os.makedirs(frames_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = os.path.join(frames_dir, f'{question}_{frame_count:06d}.png')
        cv2.imwrite(frame_filename, frame)
        frame_count += 1
    cap.release()
    print(f"All frames are extracted and saved to {frames_dir}. Total frames: {frame_count}")

def convert_video_to_frames(video_path):
    if os.path.isdir(video_path): # if video_path is a list of videos
        video = os.listdir(video_path)
    elif os.path.isfile(video_path): # else if video_path is a single video
        video = [os.path.basename(video_path)]
        video_path = os.path.dirname(video_path)     
    video.sort()
    print("start converting video to video with 16 frames from path:", video_path)
    for v in video:
        v_mp4 = v.split(".")[0] + ".mp4"
        output_path = os.path.join(os.path.dirname(video_path), "video_standard", os.path.basename(video_path))
        os.makedirs(output_path, exist_ok=True)
        convert_video(os.path.join(video_path, f"{v}"), os.path.join(output_path, f"{v_mp4}"))
    print("finish converting from path:", video_path)

    ## extract frames
    all_vid_frames_dir = os.path.join(os.path.dirname(video_path), "frames", os.path.basename(video_path))
    os.makedirs(all_vid_frames_dir, exist_ok=True)
    vid_frames_dir_list = os.listdir(all_vid_frames_dir)
    all_vid_dir = output_path
    vid_dir_list = os.listdir(all_vid_dir)
    for vid in vid_dir_list:
        name = vid.replace(".mp4","")     
        if name not in vid_frames_dir_list:
            new_frames_dir = os.path.join(all_vid_frames_dir,name)
            os.makedirs(new_frames_dir, exist_ok=True)          
            vid_path = os.path.join(all_vid_dir, vid)         
            question = name
            video2img(vid_path,new_frames_dir,question)
    print("saved frames to:", all_vid_frames_dir)
    return all_vid_frames_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-path', type=str)
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'])
    parser.add_argument('--pred-only', dest='pred_only',
                        default=True, help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', default=True, help='do not apply colorful palette')
    parser.add_argument(
        "--output_dir", "-o", type=str, default="../output_depth", help="output directory"
    )
    parser.add_argument("--read-prompt-file", type=str, default="../meta_data/spatial relationships.json")
    parser.add_argument("--video_folder", type=str, default="../video/spatial_relationships")
    
    args = parser.parse_args()
    
    margin_width = 50
    caption_height = 60
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(args.encoder)).to(DEVICE).eval()
    
    total_params = sum(param.numel() for param in depth_anything.parameters())
    print('Total parameters: {:.2f}M'.format(total_params / 1e6))
    
    transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])
    
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    with open(args.read_prompt_file,'r') as json_data:
        prompts = json.load(json_data)
        
    frame_folder = convert_video_to_frames(args.video_folder)
    videos = os.listdir(frame_folder)
    videos.sort(key=lambda x: int(x))
    
    
    for vid in videos:
        images = os.listdir(os.path.join(frame_folder,vid))
        images.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        p_num = int(vid)-1
        prompt=prompts[p_num]["prompt"]
        spatial = prompts[p_num]["spatial"] #A is on the left of B
        phrase_0 = prompts[p_num]["object_1"]
        phrase_1 = prompts[p_num]["object_2"]
        spatial = prompts[p_num]["spatial"]
        if spatial not in ["left","right","above","on","under","below","in front of","behind"]:
            print(spatial, "spatial not included!!!, index: ", vid)
            break
        
        if spatial in ["behind", "in front of"]:
        
            for frame in images:
                filename  = os.path.join(frame_folder,vid,frame)
                raw_image = cv2.imread(filename)
                image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
                
                h, w = image.shape[:2]
                
                image = transform({'image': image})['image']
                image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    depth = depth_anything(image)
                
                depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
                depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
                
                depth = depth.cpu().numpy().astype(np.uint8)
                
                if args.grayscale:
                    depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
                else:
                    depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
                
                filename = os.path.basename(filename)
                
                if args.pred_only:
                    # cv2.imwrite(os.path.join(args.outdir, filename[:filename.rfind('.')] + '_depth.png'), depth)
                    os.makedirs(os.path.join(output_dir,vid),exist_ok=True)
                    cv2.imwrite(os.path.join(output_dir,vid,frame), depth)
                    
                else:
                    split_region = np.ones((raw_image.shape[0], margin_width, 3), dtype=np.uint8) * 255
                    combined_results = cv2.hconcat([raw_image, split_region, depth])
                    
                    caption_space = np.ones((caption_height, combined_results.shape[1], 3), dtype=np.uint8) * 255
                    captions = ['Raw image', 'Depth Anything']
                    segment_width = w + margin_width
                    
                    for i, caption in enumerate(captions):
                        # Calculate text size
                        text_size = cv2.getTextSize(caption, font, font_scale, font_thickness)[0]

                        # Calculate x-coordinate to center the text
                        text_x = int((segment_width * i) + (w - text_size[0]) / 2)

                        # Add text caption
                        cv2.putText(caption_space, caption, (text_x, 40), font, font_scale, (0, 0, 0), font_thickness)
                    
                    final_result = cv2.vconcat([caption_space, combined_results])
                    
                    cv2.imwrite(os.path.join(args.outdir, filename[:filename.rfind('.')] + '_img_depth.png'), final_result)
                