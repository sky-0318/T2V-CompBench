import argparse
import torch
import csv
import json
import os
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

def extract_json(string):
    # Find the start and end positions of the JSON part
    start = string.find('{')
    end = string.rfind('}') + 1

    # Extract the JSON part from the string
    json_part = string[start:end]

    # Load the JSON part as a dictionary
    try:
        json_data = json.loads(json_part)
    except json.JSONDecodeError:
        # Handle the case when the JSON part is not valid
        print("Invalid JSON part")
        return None

    return json_data


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
    with open(args.read_prompt_file,'r') as json_data:
        prompts = json.load(json_data)
        
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, f'consistent_attr_score.csv'), 'w', newline='') as csvfile: #TODO
        # Create a CSV writer
        csv_writer = csv.writer(csvfile)
        # Write the header row
        csv_writer.writerow(["name","prompt", "Score"])
        
        grid_images = [f for f in os.listdir(video_grid_folder_prefix) if f[0].isdigit()]
        grid_images.sort(key=lambda x: int(x.split('.')[0]))#sort
        
        for i in range(len(grid_images)):
            grid_image_name = grid_images[i]
            phrases = prompts[i]["blip"] 
            this_prompt = prompts[i]["prompt"]
            phrase_1 = phrases.split(";")[0].strip()
            phrase_2 = phrases.split(";")[1].strip()
        
            image_files = [os.path.join(video_grid_folder_prefix, grid_images[i])]
            images = load_images(image_files)
            image_sizes = [x.size for x in images]
            images_tensor = process_images(  
                images,
                image_processor,
                model.config
            ).to(model.device, dtype=torch.float16)
            
            initial = "The provided image arranges key frames from a video in a grid view.  Describe the video within 20 words, carefully examining the characters or objects throughout the frames and their visible attributes."
            qs = initial
            image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            
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
                
            outputs_1 = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            conv.messages[-1][-1] = outputs_1

            qs = f"According to your previous description, please select one answer from options A1 to D1 for the first multiple choice question, and select one answer from options A2 to D2 for the second one. \n \
Question 1: \n \
A1: '{phrase_1}' is clearly portrayed throughout the frames. \n \
B1: '{phrase_1}' is present in some frames. \n \
C1: '{phrase_1}' is not correctly portrayed. \n \
D1: '{phrase_1}' is not present. \n \
Question 2: \n \
A2: '{phrase_2}' is clearly portrayed throughout the frames. \n \
B2: '{phrase_2}' is present in some frames. \n \
C2: '{phrase_2}' is not correctly portrayed. \n \
D2: '{phrase_2}' is not present. \n \
Seperate the two options by a comma and put it in JSON format with the following keys: option (e.g., A1,B2), explanation (within 20 words)."
        
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
                
            outputs_2 = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            
            json_obj = extract_json(outputs_2)
            option_value = json_obj["option"]
            options = option_value.split(",")
            
            if option_value in ["A1,A2","A2,A1"]:
                score_tmp = 5
            elif option_value in ["A1,B2","B1,A2","A2,B1","B2,A1",]:
                score_tmp = 4.5
            elif option_value in ["B1,B2","B2,B1"]:
                score_tmp = 4
            elif option_value in ["A1,C2","C1,A2","A2,C1","C2,A1"]:
                score_tmp = 3.5
            elif option_value in ["A1,D2","D1,A2","A2,D1","D2,A1"]:
                score_tmp = 3
            elif option_value in ["B1,C2","C1,B2","B2,C1","C2,B1"]:
                score_tmp = 2.8
            elif option_value in ["B1,D2","D1,B2","B2,D1","D2,B1"]:
                score_tmp = 2.5
            elif option_value in ["C1,C2","C2,C1"]:
                score_tmp = 2
            elif option_value in ["C1,D2","D1,C2","C2,D1","D2,C1"]:
                score_tmp = 1.5
            elif option_value in ["D1,D2","D2,D1"]:
                score_tmp = 1
            else:
                score_tmp = "NO WAY"
                print("no way")
                
            print("score for",grid_images[i] , score_tmp)
            grid_image_name.replace(".jpg", "")
            csv_writer.writerow([grid_image_name,this_prompt,score_tmp])
            csvfile.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/group/xihuiliu/sky/T2V-Compbench/metric/models/LLaVA/llava-v1.6-34b/")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--output-path", type=str, default="../csv_output_consistent_attr")
    parser.add_argument("--read-prompt-file", type=str, default="../meta_data/consistent attribute binding.json")
    parser.add_argument(
        "--video_grid_folder_prefix",
        type=str,
        required=True,
        help="path to video folder or certain video",
    )
    args = parser.parse_args()

    eval_model(args)
