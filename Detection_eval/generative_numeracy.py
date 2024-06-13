import argparse
import os

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


def calculate_iou(box0,box1):#xywh
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
    return IoU, IoMinA

def numeracy_judge(boxes,probs,required_num,iou_threshold):

    new_bbox = []
    new_prob = []
    for i in range(len(boxes)):
        flag = 0
        for j in range(len(new_bbox)):
            IoU, _ = calculate_iou(boxes[i], new_bbox[j])
            if IoU>iou_threshold:
                flag = 1
                if probs[i] > new_prob[j]:
                    new_bbox[-1] = boxes[i]
                    new_prob[-1] = probs[i] 
                break
        if flag == 0:
            new_bbox.append(boxes[i])
            new_prob.append(probs[i])
            
    if len(new_bbox)==required_num: 
        good_numeracy=1
    else:
        good_numeracy=0
    
    obj_json = {
        "boxes":boxes.tolist(),
        "probs":new_prob,
        "correct_num": len(new_bbox),
        "required_num":required_num,
        "good_numeracy":good_numeracy,
    }
                   
    '''
    {
        "object_1":
            {
                "boxes":[[],[],[],[]...],
                "probs":[a,a,a,a,....],
                "correct_num":n,
                "required_num": m,
                "good_numeracy": 1 or 0,
            },
        "object_2":
            {
                "boxes":[[],[],[],[]...],
                "probs":[a,a,a,a,....],
                "correct_num":n,
                "required_num": m,
                "good_numeracy": 1 or 0,
            },
    }
    '''
    return obj_json,new_bbox,new_prob
    
def numeracy_judge_cross(all_box,all_prob,nums,iou_threshold):
    
    bbox_1 = all_box[0]
    prob_1= all_prob[0]
    bbox_2 = all_box[1]
    prob_2= all_prob[1]
    new_bbox_1 = []
    new_prob_1 = []
    new_bbox_2 = []
    new_prob_2 = []
    delete_bbox_1 = []
    delete_bbox_2 = []
    
    for i in range(len(bbox_1)):
        for j in range(len(bbox_2)):
            IoU, _ = calculate_iou(bbox_2[j], bbox_1[i])
            if IoU>iou_threshold:
                if prob_2[j] > prob_1[i]:
                   delete_bbox_1.append(i)
                else:
                   delete_bbox_2.append(j)
    
    unique_delete_bbox_1 = list(set(delete_bbox_1))    
    unique_delete_bbox_2 = list(set(delete_bbox_2))    
    new_bbox_1 = [element for index, element in enumerate(bbox_1) if index not in unique_delete_bbox_1]
    new_prob_1 = [element for index, element in enumerate(prob_1) if index not in unique_delete_bbox_1]
    new_bbox_2 = [element for index, element in enumerate(bbox_2) if index not in unique_delete_bbox_2]
    new_prob_2 = [element for index, element in enumerate(prob_2) if index not in unique_delete_bbox_2]
    
       
           
    if len(new_bbox_1)==nums[0]: 
        good_numeracy=1
    else:
        good_numeracy=0

    obj1_json = {
        "boxes":[tensor_list.tolist() for tensor_list in bbox_1],
        "probs":new_prob_1,
        "correct_num": len(new_bbox_1),
        "required_num":nums[0],
        "good_numeracy":good_numeracy,
        "delete_1": unique_delete_bbox_1
    }
    
    if len(new_bbox_2)==nums[1]: 
        good_numeracy=1
    else:
        good_numeracy=0

    obj2_json = {
        "boxes":[tensor_list.tolist() for tensor_list in bbox_2],
        "probs":new_prob_2,
        "correct_num": len(new_bbox_2),
        "required_num":nums[1],
        "good_numeracy":good_numeracy,
        "delete_2": unique_delete_bbox_2
    }
    
    obj_json_2 = {}
    obj_json_2["object_1"] = obj1_json  
    obj_json_2["object_2"] = obj2_json  
    return obj_json_2


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

def combine_frame(input_csv, output_csv):
    with open(input_csv, 'r') as file:
        reader = csv.reader(file)
        lines = list(reader)
        
        batch_size = 16
        num_vid = (len(lines)-1) / batch_size
        if num_vid!=int(num_vid):
            print("error: number of lines WRONG")
        
        score_vid_1 = [] 
        id = []
        score_frame_1 = []
    
        for i in range(int(num_vid)):
            batch = lines[i * batch_size + 1 : (i + 1) * batch_size + 1]
            # Process the batch of lines
            frame_score_1 = []
            
            id.append(batch[0][0])
            for line in batch:
                frame_score_1.append(float(line[-1]))
                        
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounding DINO example", add_help=True)
    parser.add_argument("--config_file", "-c", type=str, default="../groundingdino/config/GroundingDINO_SwinT_OGC.py", help="path to config file")
    parser.add_argument(
        "--checkpoint_path", "-p", type=str, default="../weights/groundingdino_swint_ogc.pth", help="path to checkpoint file"
    )
    parser.add_argument(
        "--output_dir", "-o", type=str, default="../output_generative_numeracy", help="output directory"
    )
    parser.add_argument("--csv_dir", default="../csv_dir/generative_numeracy", type=str)
    parser.add_argument("--iou_threshold", default=0.9, type=float)

    parser.add_argument("--box_threshold", type=float, default=0.4, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--token_spans", type=str, default=None, help=
                        "The positions of start and end positions of phrases of interest. \
                        For example, a caption is 'a cat and a dog', \
                        if you would like to detect 'cat', the token_spans should be '[[[2, 5]], ]', since 'a cat and a dog'[2:5] is 'cat'. \
                        if you would like to detect 'a cat', the token_spans should be '[[[0, 1], [2, 5]], ]', since 'a cat and a dog'[0:1] is 'a', and 'a cat and a dog'[2:5] is 'cat'. \
                        ")

    parser.add_argument("--cpu-only", action="store_true", help="running on cpu only!, default=False")
    parser.add_argument("--read-prompt-file", type=str, default="../meta_data/generative numeracy.json")
    parser.add_argument("--video_folder", type=str, default="../video/generative_numeracy")
    
    args = parser.parse_args()

    # cfg
    config_file = args.config_file  # change the path of the model config file
    checkpoint_path = args.checkpoint_path  # change the path of the model
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    token_spans = args.token_spans
    iou_threshold = args.iou_threshold
    # load model
    model = load_model(config_file, checkpoint_path, cpu_only=args.cpu_only)

    with open(args.read_prompt_file,'r') as json_data:
        prompts = json.load(json_data)

    # make dir
    os.makedirs(output_dir, exist_ok=True)
    

    
    frame_folder = convert_video_to_frames(args.video_folder)
    videos = os.listdir(frame_folder)
    videos.sort(key=lambda x: int(x))#sort
    
    os.makedirs(args.csv_dir,exist_ok=True)
    with open(f'{args.csv_dir}/numeracy_frame.csv', 'w', newline='') as csvfile:
        # Create a CSV writer
        csv_writer = csv.writer(csvfile)
        # Write the header row (optional)
        csv_writer.writerow(["video_name","image_name","prompt","objects","numbers","objs_json","objs_json_2","correct_num_1","correct_num_2","good_numeracy_1","good_numeracy_2","score"])
        
        for i in range(len(videos)):
            os.makedirs(os.path.join(output_dir,videos[i]),exist_ok=True)
            video_path = os.path.join(frame_folder,videos[i])
            images = os.listdir(video_path)
            images.sort(key=lambda x: int(x.split("_")[-1].split('.')[0]))#sort
            
            index = int(videos[i])
            prompt=prompts[index-1]["prompt"]
            objects = prompts[index-1]["objects"]#A,B,C,D
            numbers =  prompts[index-1]["numbers"]#x1,x2,x3,x4
            objs = objects.split(",")
            nums = numbers.split(",")
            if len(objs) != len(nums):
                print("video ",i," parse wrong")
                break
            for j in range(len(objs)):
                objs[j].strip()
                try:
                    nums[j] = int(nums[j])
                except:
                    print("object number not int")
            
            for image_name in images:
                # load image
                num = image_name.split(".")[0]
                image_path = os.path.join(frame_folder,videos[i],image_name)
                image_pil, image = load_image(image_path)
                
                if token_spans is not None:
                    text_threshold = None
                    print("Using token_spans. Set the text_threshold to None.")

                # run model
                objs_json = {}
                image_with_box = image_pil
                all_box = []
                all_prob = []
                for j in range(len(objs)):
                
                    boxes_filt_0, pred_phrases_0, prob_0 = get_grounding_output(
                        model, image, objs[j], box_threshold, text_threshold, cpu_only=args.cpu_only, token_spans=eval(f"{token_spans}")
                    )
                    size = image_pil.size
                    
                    obj_boxes_json,new_box,new_prob = numeracy_judge(boxes_filt_0,prob_0,nums[j],iou_threshold)
                    objs_json[f"object_{j}"] = obj_boxes_json
                    all_box.append(new_box)
                    all_prob.append(new_prob)
                   
                    pred_dict_0 = {
                        "boxes": boxes_filt_0,
                        "size": [size[1], size[0]],  # H,W
                        "labels": pred_phrases_0,
                    }
                    image_with_box = plot_boxes_to_image(image_with_box, pred_dict_0)[0]
                
                if len(objs)>1:
                    obj_boxes_json_2 = numeracy_judge_cross(all_box,all_prob,nums,iou_threshold)
                else:
                    obj_boxes_json_2 = ""
            
                if obj_boxes_json_2!="":
                    correct_num_1 = obj_boxes_json_2["object_1"]["correct_num"]
                    correct_num_2 = obj_boxes_json_2["object_2"]["correct_num"]
                    good_numeracy_1 = obj_boxes_json_2["object_1"]["good_numeracy"]
                    good_numeracy_2 = obj_boxes_json_2["object_2"]["good_numeracy"]
                    a = 0
                    if good_numeracy_1==1:
                        a+=0.5
                    if good_numeracy_2==1:
                        a+=0.5
                        
                else:
                    correct_num_1 = objs_json["object_0"]["correct_num"]
                    correct_num_2 = ""
                    good_numeracy_1 = objs_json["object_0"]["good_numeracy"]
                    good_numeracy_2 = ""
                    a = 0
                    if good_numeracy_1==1:
                        a+=1
                csv_writer.writerow([videos[i],image_name, prompt, objs,nums,objs_json,obj_boxes_json_2,correct_num_1,correct_num_2,good_numeracy_1,good_numeracy_2,a])
                image_with_box.save(os.path.join(output_dir,videos[i],f"{num}.jpg"))
                
    combine_frame(f'{args.csv_dir}/numeracy_frame.csv', f'{args.csv_dir}/numeracy_video.csv')  