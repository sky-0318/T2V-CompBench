import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span

from tqdm import tqdm
import json
import csv
import cv2
from torchvision.io import write_video


def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    # draw boxes and masks
    for box, label in zip(boxes, labels):
        # from 0..1 to 0..W, 0..H
        box = box * torch.Tensor([W, H, W, H])
        xc = int(box[0])
        yc = int(box[1])
        s = 3
        # from xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        # random color
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        # draw
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
        # draw.text((x0, y0), str(label), fill=color)

        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)
        # bbox = draw.textbbox((x0, y0), str(label))
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white")
        draw.ellipse((xc-s,yc-s,xc+s,yc+s), fill=color)

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    return image_pil, mask


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold=None, with_logits=True, cpu_only=False, token_spans=None):
    assert text_threshold is not None or token_spans is not None, "text_threshould and token_spans should not be None at the same time!"
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    device = "cuda" if not cpu_only else "cpu"
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4)

    # filter output
    if token_spans is None:
        logits_filt = logits.cpu().clone()
        boxes_filt = boxes.cpu().clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        #sky
        prob = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)
            #sky
            prob.append(logit.max().item())
            
    else:
        # given-phrase mode
        positive_maps = create_positive_map_from_span(
            model.tokenizer(text_prompt),
            token_span=token_spans
        ).to(image.device) # n_phrase, 256

        logits_for_phrases = positive_maps @ logits.T # n_phrase, nq
        all_logits = []
        all_phrases = []
        all_boxes = []
        for (token_span, logit_phr) in zip(token_spans, logits_for_phrases):
            # get phrase
            phrase = ' '.join([caption[_s:_e] for (_s, _e) in token_span])
            # get mask
            filt_mask = logit_phr > box_threshold
            # filt box
            all_boxes.append(boxes[filt_mask])
            # filt logits
            all_logits.append(logit_phr[filt_mask])
            if with_logits:
                logit_phr_num = logit_phr[filt_mask]
                all_phrases.extend([phrase + f"({str(logit.item())[:4]})" for logit in logit_phr_num])
            else:
                all_phrases.extend([phrase for _ in range(len(filt_mask))])
        boxes_filt = torch.cat(all_boxes, dim=0).cpu()
        pred_phrases = all_phrases


    return boxes_filt, pred_phrases, prob

def pick_1_box(boxes,probs,size):
    H = size[1]
    W = size[0]
    if len(boxes)>1:
        m = max(probs)
        ind = probs.index(m)
        box = boxes[ind]
        box = box * torch.Tensor([W, H, W, H])
    elif len(boxes)==1:
        box = boxes[0]
        m = probs[0]
        box = box * torch.Tensor([W, H, W, H])
    else: #empty
        box = []
        m=0
    return box, m

def clean_boxes(boxes,size):
    H = size[1]
    W = size[0]
    clean_boxes = []
    m=1
    for i in range(len(boxes)):
        box = boxes[i]
        box = box * torch.Tensor([W, H, W, H])
        clean_boxes.append(box)
    if len(clean_boxes)==0:
        m=0
    return clean_boxes, m


def spatial_judge(box0,box1,spatial):
    #box:[xc,yc,w,h]
    good_spatial=0
    IoU = 0
    IoMinA = 0
    XIoMinX = 0
    YIoMinY = 0
    
    if len(box0)!=0 and len(box1)!=0:
        centre_0 = [box0[0].item(),box0[1].item()]
        centre_1 = [box1[0].item(),box1[1].item()]
        dw=centre_0[0]-centre_1[0] #x0-x1
        dh=centre_0[1]-centre_1[1] #y0-y1
        
        # Calculate IoU
        # from xywh to xyxy
        [box0_xmin, box0_ymin] = box0[:2] - box0[2:] / 2
        [box0_xmax, box0_ymax] = box0[:2] + box0[2:] / 2
        [box1_xmin, box1_ymin] = box1[:2] - box1[2:] / 2
        [box1_xmax, box1_ymax] = box1[:2] + box1[2:] / 2
       
        x_overlap = max(0, min(box0_xmax,box1_xmax) - max(box0_xmin,box1_xmin))
        y_overlap = max(0, min(box0_ymax,box1_ymax) - max(box0_ymin,box1_ymin))
        
        intersection = x_overlap * y_overlap
        box0_area = box0[2]*box0[3]
        box1_area = box1[2]*box1[3]
        union = box0_area + box1_area - intersection
        
        IoU = intersection / union #intersection over union
        IoMinA = intersection / min(box0_area,box1_area) #intersection over the smaller box area
        XIoMinX = x_overlap / min(box0[2],box1[2])#intersection of x over the min w
        YIoMinY = y_overlap / min(box0[3],box1[3])#intersection of x over the min w
        IoU = IoU.item()
        IoMinA = IoMinA.item()
        XIoMinX = XIoMinX.item()
        YIoMinY = YIoMinY.item()
        
        if dw<0.0 and abs(dw)>abs(dh):
            correct_spatial="left"
        elif dw>0.0 and abs(dw)>abs(dh):
            correct_spatial="right"
        elif dh<0.0 and abs(dh)>abs(dw):
            correct_spatial="above"
        elif dh>0.0 and abs(dh)>abs(dw):
            correct_spatial="under"
        else:
            correct_spatial=""
            
        if correct_spatial==spatial:
            good_spatial=1
        elif spatial=="on" and correct_spatial=="above":
            good_spatial=1
        elif spatial=="below" and correct_spatial=="under":
            good_spatial=1
        else:
            good_spatial=0
            
    elif len(box0)==0 and len(box1)!=0:
        correct_spatial = "no object_1"
        good_spatial = 0
        centre_0 = None
        centre_1 = [box1[0].item(),box1[1].item()]
    elif len(box1)==0 and len(box0)!=0:   
        correct_spatial = "no object_2"
        good_spatial = 0 
        centre_1 = None
        centre_0 = [box0[0].item(),box0[1].item()]
    else: #both not detected'
        correct_spatial = "neither",
        good_spatial = 0 
        centre_1 = None
        centre_0 = None         
   
    return  correct_spatial, good_spatial, centre_0, centre_1, IoU, IoMinA, XIoMinX, YIoMinY

def pick_max(total_score_1_list,record_all_good_spatial):
    max1 = max(total_score_1_list)
    ind1 = total_score_1_list.index(max1)
    best_box_1 = record_all_good_spatial[ind1]
    score_1 = best_box_1["spatial_score_1"]
    return score_1

def combine_frame(input_csv, output_csv):
    with open(input_csv, 'r') as file:
        reader = csv.reader(file)
        lines = list(reader)
        
        batch_size = 16
        num_vid = (len(lines)-1) / batch_size
        if num_vid!=int(num_vid):
            print("error: number of lines WRONG")
        
        score_vid_1 = [] 
        score_frame_1 = []
      
        for i in range(int(num_vid)):
            batch = lines[i * batch_size + 1 : (i + 1) * batch_size + 1]
            # Process the batch of lines
            frame_score_1 = []
           
            id.append(batch[0][0])
            for line in batch:
                tmp1 = float(line[-1])
             
                if tmp1<-1: #=-2
                    tmp1 = 0
                elif tmp1<0: #=-1
                    tmp1=0.2
                elif tmp1==0: #=0
                    tmp1=0.4
                elif tmp1>0: 
                    tmp1=tmp1*0.6 + 0.4
        
                frame_score_1.append(tmp1)
            
            score_vid_1.append(sum(frame_score_1)/16)
            score_frame_1.append(frame_score_1)
    
    score_vid_1 = ["Score_1"]+ score_vid_1
    
    id = ["id"]+id
    score_frame_1 = ["Score_frame_1"] + score_frame_1
    
    if len(score_vid_1) != len(score_frame_1) !=len(id):
        print("counting error")
    
    with open(output_csv, 'w') as output_file:
        writer = csv.writer(output_file)
        for i in range(len(id)):
            # Append data to the end of each row
            row = [id[i],score_frame_1[i],score_vid_1[i]]
            # Write the modified row to the new file
            writer.writerow(row)
            

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
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounding DINO example", add_help=True)
    parser.add_argument("--config_file", "-c", type=str, default="../groundingdino/config/GroundingDINO_SwinT_OGC.py", help="path to config file")
    parser.add_argument(
        "--checkpoint_path", "-p", type=str, default="../weights/groundingdino_swint_ogc.pth", help="path to checkpoint file"
    )
    parser.add_argument(
        "--output_dir", "-o", type=str, default="../output_spatial_relationships_2d", help="output directory"
    )
    parser.add_argument("--csv_dir", default="../csv_dir/spatial_relationships_2d", type=str)
    

    parser.add_argument("--box_threshold", type=float, default=0.35, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--token_spans", type=str, default=None, help=
                        "The positions of start and end positions of phrases of interest. \
                        For example, a caption is 'a cat and a dog', \
                        if you would like to detect 'cat', the token_spans should be '[[[2, 5]], ]', since 'a cat and a dog'[2:5] is 'cat'. \
                        if you would like to detect 'a cat', the token_spans should be '[[[0, 1], [2, 5]], ]', since 'a cat and a dog'[0:1] is 'a', and 'a cat and a dog'[2:5] is 'cat'. \
                        ")

    parser.add_argument("--cpu-only", action="store_true", help="running on cpu only!, default=False")
    parser.add_argument("--read-prompt-file", type=str, default="../meta_data/spatial relationships.json")
    parser.add_argument("--video_folder", type=str, default="../video/spatial_relationships")
    args = parser.parse_args()

    # cfg
    config_file = args.config_file  
    checkpoint_path = args.checkpoint_path  
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    token_spans = args.token_spans

    
    # load model
    model = load_model(config_file, checkpoint_path, cpu_only=args.cpu_only)

    with open(args.read_prompt_file,'r') as json_data:
        prompts = json.load(json_data)

    # make dir
    os.makedirs(output_dir, exist_ok=True)
    
    frame_folder = convert_video_to_frames(args.video_folder)

    videos = os.listdir(frame_folder)
    videos.sort(key=lambda x: int(x))
    
    os.makedirs(args.csv_dir,exist_ok=True)
    with open(f'{args.csv_dir}/2d_frame.csv', 'w', newline='') as csvfile:
        # Create a CSV writer
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["video_name","image_name","prompt","object_1","object_2","score"])
                            
        for i in range(len(videos)): 
            p_num = int(videos[i])-1
            prompt=prompts[p_num]["prompt"]
            spatial = prompts[p_num]["spatial"] #A is on the left of B
            phrase_0 = prompts[p_num]["object_1"]
            phrase_1 = prompts[p_num]["object_2"]

            if spatial not in ["left","right","above","on","under","below","in front of","behind"]:
                print(spatial, "spatial not included!!!, index: ", videos[i])
                break
                
            if spatial in ["left","right","above","on","under","below"]:  
                os.makedirs(os.path.join(output_dir,videos[i]),exist_ok=True)
                
                video_path = os.path.join(frame_folder,videos[i])
                images = os.listdir(video_path)
                images.sort(key=lambda x: int(x.split("_")[-1].split('.')[0]))#sort
                
                
                for image_name in images:
                    image_path = os.path.join(frame_folder,videos[i],image_name)
                    image_pil, image = load_image(image_path)
                    
                    if token_spans is not None:
                        text_threshold = None
                        print("Using token_spans. Set the text_threshold to None.")

                    # run model
                    boxes_filt_0, pred_phrases_0, prob_0 = get_grounding_output(
                        model, image, phrase_0, box_threshold, text_threshold, cpu_only=args.cpu_only, token_spans=eval(f"{token_spans}")
                    )
                    boxes_filt_1, pred_phrases_1, prob_1 = get_grounding_output(
                        model, image, phrase_1, box_threshold, text_threshold, cpu_only=args.cpu_only, token_spans=eval(f"{token_spans}")
                    )
                    size = image_pil.size
                    
                    clean_boxes_0, m0 = clean_boxes(boxes_filt_0,size)
                    clean_boxes_1, m1 = clean_boxes(boxes_filt_1,size)
                    
                    
                    if m0!=0 and m1!=0:
                        record_all_good_spatial = []
                        for ii in range(len(clean_boxes_0)):
                            for jj in range(len(clean_boxes_1)):
                                correct_spatial, good_spatial,centre_0,centre_1, IoU, IoMinA, XIoMinX, YIoMinY = spatial_judge(clean_boxes_0[ii],clean_boxes_1[jj],spatial)
                                if good_spatial==1:
                                    spatial_score_1 = good_spatial * (1-IoU)
                                        
                                    prob_score_A = 0.5*prob_0[ii]+0.5*prob_1[jj]
                                 
                                    total_score_1 = 0.5*spatial_score_1 + 0.5*prob_score_A  
                                    
                                    info = {}
                                    info["name"]=f"{ii}_{jj}"
                                    info["box0"]=clean_boxes_0[ii]
                                    info["box1"]=clean_boxes_1[jj]
                                    info["total_score_1"] = total_score_1
                                    info["spatial_score_1"] = spatial_score_1
                                
                                    record_all_good_spatial.append(info)
                                
                        if len(record_all_good_spatial) !=0: 
                            total_score_1_list = [] 
                            
                            for candidate_box in record_all_good_spatial:
                                total_score_1_list.append(candidate_box["total_score_1"])  
                            
                            score_1 = pick_max(total_score_1_list,record_all_good_spatial)
                        
                        else:        
                            score_1 = 0                                   
                  
                    elif (m0==0 and m1!=0) or (m0!=0 and m1==0):
                        score_1 = -1
                    elif m0==0 and m1==0:
                        score_1 = -2     
                    
                    
                    csv_writer.writerow([videos[i],image_name, prompt, m0,m1, score_1])
                    csvfile.flush() 
                                  
                    # visualize pred
                    pred_dict_0 = {
                        "boxes": boxes_filt_0,
                        "size": [size[1], size[0]],  # H,W
                        "labels": pred_phrases_0,
                    }
                    pred_dict_1 = {
                        "boxes": boxes_filt_1,
                        "size": [size[1], size[0]],  # H,W
                        "labels": pred_phrases_1,
                    }
                
                    
                    # import ipdb; ipdb.set_trace()
                    image_with_box = plot_boxes_to_image(image_pil, pred_dict_0)[0]
                    image_with_box_1 = plot_boxes_to_image(image_with_box, pred_dict_1)[0]
                    os.makedirs(os.path.join(output_dir, videos[i]),exist_ok=True)
                    image_with_box_1.save(os.path.join(output_dir, videos[i],image_name))
        
        combine_frame(f'{args.csv_dir}/2d_frame.csv', f'{args.csv_dir}/2d_video.csv')    