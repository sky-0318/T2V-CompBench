import argparse
import torch
import csv
import json
import os
from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re

import cv2
import moviepy.editor as mp
import numpy as np
from torchvision.io import write_video

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)


def extract_frames(video_path, num_frames=16):
    # Capture the video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate frame indices to extract
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

    # Load the video
    cap = cv2.VideoCapture(video_path)

    # Initialize frame count
    frame_count = 0

    # Loop through each frame in the video
    while True:
        # Read frame from the video
        ret, frame = cap.read()

        # Check if frame is read correctly
        if not ret:
            break

        # Define the filename for each frame
        frame_filename = os.path.join(frames_dir, f'{question}_{frame_count:06d}.png')

        # Save the frame as a PNG file
        cv2.imwrite(frame_filename, frame)
        
        # Increment the frame count
        frame_count += 1

    # Release the video capture object
    cap.release()

    print(f"All frames are extracted and saved to {frames_dir}. Total frames: {frame_count}")


def merge_grid(image_folder, image_path_list):
    # Open the images
    os.path.join(image_folder,image_path_list[0])
    image1 = Image.open(os.path.join(image_folder,image_path_list[0]))
    image2 = Image.open(os.path.join(image_folder,image_path_list[1]))
    image3 = Image.open(os.path.join(image_folder,image_path_list[2]))
    image4 = Image.open(os.path.join(image_folder,image_path_list[3]))
    image5 = Image.open(os.path.join(image_folder,image_path_list[4]))
    image6 = Image.open(os.path.join(image_folder,image_path_list[5]))


    # Create a new blank image with the desired size
    grid_width = 2 * image1.width
    grid_height = 3 * image1.height
    grid_image = Image.new('RGB', (grid_width, grid_height))

    # Paste the images into the grid
    grid_image.paste(image1, (0, 0))
    grid_image.paste(image2, (image1.width, 0))
    grid_image.paste(image3, (0, image1.height))
    grid_image.paste(image4, (image1.width, image1.height))
    grid_image.paste(image5, (0, 2*image1.height))
    grid_image.paste(image6, (image1.width, 2*image1.height))

    # Save the merged image
    return grid_image 


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
        
    # get grid
    frame_folder = all_vid_frames_dir
    os.makedirs(frame_folder+"_grid", exist_ok=True)

    grid_folder = frame_folder+"_grid"

    for folder in os.listdir(frame_folder):
        video_frames = os.listdir(os.path.join(frame_folder,folder))
        
        video_frames.sort(key=lambda x: int(x.split("_")[-1].split('.')[0]))#sort
        
        grid = [video_frames[0],video_frames[3],video_frames[6],video_frames[9],video_frames[12],video_frames[15]]
        grid_image = merge_grid(os.path.join(frame_folder,folder), grid)
        grid_image.save(grid_folder+"/"+folder+".jpg")
    print("saved grid images to:", grid_folder)

def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def eval_model(args):

    # preprocess video, convert it into video with 16 frames, frames, and grid
    convert_video_to_frames(args.video_grid_folder_prefix)
    if os.path.isdir(args.video_grid_folder_prefix): # repo with a list of videos
        video_grid_folder_prefix = os.path.join(os.path.dirname(args.video_grid_folder_prefix), "frames", os.path.basename(args.video_grid_folder_prefix) + "_grid")
    elif os.path.isfile(args.video_grid_folder_prefix): # a single video
        video_path_tmp = os.path.dirname(args.video_grid_folder_prefix)
        video_grid_folder_prefix = os.path.join(os.path.dirname(video_path_tmp), "frames", os.path.basename(video_path_tmp) + "_grid")
        
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )


    with open(args.read_prompt_file,'r') as f: 
        prompts = f.read().splitlines()

    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, f'interaction_score.csv'), 'w', newline='') as csvfile: #TODO
        # Create a CSV writer
        csv_writer = csv.writer(csvfile)
        # Write the header row (optional)
        initial = "The provided image arranges key frames from a video in a grid view. Describe the video within 20 words, focusing on the interactions between characters or objects visible throughout the frames."
        
        csv_writer.writerow(["grid_image_name","prompt", 'Score'])
        
        grid_images = [f for f in os.listdir(video_grid_folder_prefix) if f[0].isdigit()]
   
        grid_images.sort(key=lambda x: int(x.split('.')[0]))#sort
        

        for i in range(len(grid_images)):
            out = []
            question = []
            grid_image_name = grid_images[i]
            score = []
            score_total = 0
            
            this_prompt = prompts[i]
            
            image_file = [os.path.join(video_grid_folder_prefix,grid_images[i])]
            images = load_images(image_file)
            image_sizes = [x.size for x in images] 
            images_tensor = process_images(  
                images,
                image_processor,
                model.config
            ).to(model.device, dtype=torch.float16)
        
            image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            qs = initial
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
            conv_mode = "chatml_direct"
            args.conv_mode = conv_mode
            conv = conv_templates[args.conv_mode].copy()

            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = (
                tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                .unsqueeze(0)
                .cuda()
            )
    
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=images_tensor,
                    image_sizes=image_sizes,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature, #0.2
                    top_p=args.top_p,
                    num_beams=args.num_beams, #1
                    max_new_tokens=args.max_new_tokens, #512
                    use_cache=True,
                )
            
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            out.append(outputs) #out[0]
            conv.messages[-1][-1] = outputs

            image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN

            qs = f"According to the video and your previous answer, evaluate if the text \'{this_prompt}\' is correctly portrayed in the video. \
                  Assign a score from 1 to 5 according the criteria: \
                    5: All the objects involved in the interaction are present, and the interaction is depicted correctly. \
                    4: All the objects involved in the interaction are present, and the interaction is almost correct. \
                    3: All the objects involved in the interaction are present, but the interaction is not correct. \
                    2: Some of the objects involved in the interaction are missing. \
                    1: None of the objects involved in the interaction are present. \
                    Provide your analysis and explanation in JSON format with the following keys: score (e.g., 2), explanation (within 20 words)."
            
            
            question.append(qs)

            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = (
                tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                .unsqueeze(0)
                .cuda()
            )

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=images_tensor,
                    image_sizes=image_sizes,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature, #0.2
                    top_p=args.top_p,
                    num_beams=args.num_beams, #1
                    max_new_tokens=args.max_new_tokens, #512
                    use_cache=True,
                )
                
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            out.append(outputs) #out[1]
            conv.messages[-1][-1] = outputs
            
            # get score from outputs
            pattern = r'"score":\s*(\d+),'
            match = re.search(pattern, outputs)

            if match:
                score_tmp = int(match.group(1))
            else:
                print('No score found')
            score.append(score_tmp)
                

                
            
            if args.forget: 
                conv.messages.pop()
                conv.messages.pop()
                conv.messages.pop()
                conv.messages.pop()
            
            # calculate score_total

            score_total = score[0]
            print("score for",grid_images[i] , score_total)

            csv_writer.writerow([grid_image_name, this_prompt, score_total])
            # update csv
            csvfile.flush()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/group/xihuiliu/sky/T2V-Compbench/metric/models/LLaVA/llava-v1.6-34b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)

    parser.add_argument("--output-path", type=str, default="../csv_output_interaction")
    parser.add_argument("--read-prompt-file", type=str, default="/group/xihuiliu/sky/T2V-Compbench/prompts/output_prompt_6_2.txt")

    parser.add_argument(
        "--video_grid_folder_prefix",
        type=str,
        # default = "/group/xihuiliu/sky/T2V-Compbench/models/ZeroScope/t2v_5_2_test/t2v_5_2", # video repo path
        default = "/group/xihuiliu/sky/T2V-Compbench/models/ZeroScope/t2v_2_2_test/t2v_2_2_single/0001.mp4", # single video        help="path to video grid folder",
    )
    parser.add_argument("--forget", type=bool, default=False, help="if forget, dispose last 2 question's answer")
    args = parser.parse_args()

    eval_model(args)
