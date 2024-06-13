import argparse
import os

import numpy as np
import json
import torch
from PIL import Image
import csv

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import (
    sam_model_registry,
    sam_hq_model_registry, 
    SamPredictor
)
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from torchvision.io import write_video


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


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    #sky
    logits_list = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)
        logits_list.append(logit.max().item())

    return boxes_filt, pred_phrases, logits_list

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        # color = np.array([30/255, 144/255, 255/255, 0.6])
        #sky
        color = np.array([255/255, 255/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    ax.text(x0, y0, label)


def save_mask_data(output_dir, mask_list, box_list, label_list):
    # value = 0  # 0 for background
    value=1

    mask_img = torch.ones(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        # mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
        mask_img[mask.cpu().numpy()[0] == True] = value - 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy(),cmap='gray')
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'mask_background.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)
    
    json_data = [{
        'value': value,
        'label': 'background'
    }]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1] 
        json_data.append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, 'mask.json'), 'w') as f:
        json.dump(json_data, f)

def save_mask_foreground(output_dir, mask,obj_prompt):
    # value = 0  # 0 for background
    value=0

    mask_img = torch.zeros(mask.shape[-2:])
    mask_img[mask.cpu().numpy()[0] == True] = value + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy(),cmap='gray')
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, f'mask_foreground_{obj_prompt}.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)
    
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


def intersection_judge(box0,box1):
    #box:[xc,yc,w,h]

    IoU = 0
    IoMinA = 0
    XIoMinX = 0
    YIoMinY = 0
    
    if not len(box0)!=0 and len(box1)!=0:
        print("NO WAY")
        
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
    
    return  intersection, IoU, IoMinA, XIoMinX, YIoMinY

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


def numeracy_judge_cross(all_box,all_prob,iou_threshold=0.95):
    
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
    
    if len(new_bbox_1)!=0:
        new_bbox_1 = torch.stack(new_bbox_1, dim=0)
    else:
        new_bbox_1 = torch.tensor([])
    if len(new_bbox_2)!=0:
        new_bbox_2 = torch.stack(new_bbox_2, dim=0)
    else:
        new_bbox_2 = torch.tensor([])

    return new_bbox_1,new_prob_1,new_bbox_2,new_prob_2


def pick_max(total_score_1_list,record_all_good_spatial):
    max1 = max(total_score_1_list)
    ind1 = total_score_1_list.index(max1)
    best_box_1 = record_all_good_spatial[ind1]
    score_1 = best_box_1["spatial_score_1"]
    return score_1


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
                my_score_1 = float(line[-1])
               
                if my_score_1<-1: #=-2
                    my_score_1 = 0
                elif my_score_1<0: #=-1
                    my_score_1 = 0.2
                elif my_score_1 == 0: #=0
                    my_score_1 = 0.4
                elif my_score_1 > 0:
                    my_score_1 = (my_score_1*0.6) + 0.4
                frame_score_1.append(my_score_1)
                
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

    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str, default="./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", help="path to config file")
    parser.add_argument(
        "--grounded_checkpoint", type=str, default="./groundingdino_swint_ogc.pth", help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_version", type=str, default="vit_h", required=False, help="SAM ViT version: vit_b / vit_l / vit_h"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, default="./sam_vit_h_4b8939.pth", help="path to sam checkpoint file"
    )
    parser.add_argument(
        "--sam_hq_checkpoint", type=str, default=None, help="path to sam-hq checkpoint file"
    )
    parser.add_argument(
        "--use_sam_hq", action="store_true", help="using sam-hq for prediction"
    )
    parser.add_argument("--input_image", type=str, default="/group/xihuiliu/sky/T2V-Compbench/metric/models/Grounded-Segment-Anything/assets/demo1.jpg", help="path to image file")
    parser.add_argument("--text_prompt", type=str, default="bear", help="text prompt")
    
    parser.add_argument(
        "--output_dir", "-o", type=str, default="./output_spatial_relationships_3d", help="output directory"
    )
    parser.add_argument("--box_threshold", type=float, default=0.35, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--device", type=str, default="cuda", help="running on cpu only!, default=False")
    
    parser.add_argument("--depth_folder", type=str, default="../output_depth")
    parser.add_argument("--read-prompt-file", type=str, default="../meta_data/spatial relationships.json")
    parser.add_argument("--video_folder", type=str, default="../video/spatial_relationships")
    parser.add_argument("--frame_folder", type=str, default="")
    parser.add_argument("--csv_dir", default="../csv_dir/spatial_relationships_3d", type=str)
    args = parser.parse_args()

    # cfg
    config_file = args.config  # change the path of the model config file
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_version = args.sam_version
    sam_checkpoint = args.sam_checkpoint
    sam_hq_checkpoint = args.sam_hq_checkpoint
    use_sam_hq = args.use_sam_hq
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    device = args.device
   
    # make dir
    os.makedirs(output_dir, exist_ok=True)
    
    # load model
    model = load_model(config_file, grounded_checkpoint, device=device)
    # initialize SAM
    if use_sam_hq:
        predictor = SamPredictor(sam_hq_model_registry[sam_version](checkpoint=sam_hq_checkpoint).to(device))
    else:
        predictor = SamPredictor(sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device))
        
        
    depth_folder = args.depth_folder
    
    
    with open(args.read_prompt_file,'r') as json_data:
        prompts = json.load(json_data)
    
    if args.frame_folder == "":    
        frame_folder = convert_video_to_frames(args.video_folder)
    else: 
        frame_folder = args.frame_folder
        
    videos = os.listdir(frame_folder)
    videos.sort(key=lambda x: int(x))

    
    os.makedirs(args.csv_dir,exist_ok=True)
    with open(f'{args.csv_dir}/3d_frame.csv', 'w', newline='') as csvfile:
        # Create a CSV writer
        csv_writer = csv.writer(csvfile)
        # Write the header row (optional)
        csv_writer.writerow(["video_name","image_name","prompt","object_1","object_2","score_1"])
                            
        for k in range(len(videos)):
            num = int(videos[k])-1
            spatial = prompts[num]["spatial"]
            if spatial in ["in front of","behind"]:
                prompt=prompts[num]["prompt"]
                object_1 = prompts[num]["object_1"] #A is on the left of B
                object_2 = prompts[num]["object_2"]
            
                images = os.listdir(os.path.join(frame_folder,videos[k]))
                images.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
                
                for image_name in images:
                    image_path = os.path.join(frame_folder,videos[k],image_name)
                    
                    # load image
                    image_pil, image_loded = load_image(image_path)
                    
                    depth_path = os.path.join(depth_folder,videos[k],image_name)
                    
                    boxes_filt_0, pred_phrases_0, probs_0 = get_grounding_output(
                        model, image_loded, object_1, box_threshold, text_threshold, device=device
                    )
                    boxes_filt_1, pred_phrases_1, probs_1 = get_grounding_output(
                        model, image_loded, object_2, box_threshold, text_threshold, device=device
                    )
                    size = image_pil.size
                    
                    all_box = [boxes_filt_0,boxes_filt_1]
                    all_prob = [probs_0,probs_1]
                    boxes_filt_0,probs_0,boxes_filt_1,probs_1 = numeracy_judge_cross(all_box,all_prob)
                    
                    clean_boxes_0, m0 = clean_boxes(boxes_filt_0,size)
                    clean_boxes_1, m1 = clean_boxes(boxes_filt_1,size)
                    
                    #sam
                    image = cv2.imread(image_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    predictor.set_image(image)
                                            
                    H, W = size[1], size[0]
                    for i in range(boxes_filt_0.size(0)):
                        boxes_filt_0[i] = boxes_filt_0[i] * torch.Tensor([W, H, W, H])
                        boxes_filt_0[i][:2] -= boxes_filt_0[i][2:] / 2
                        boxes_filt_0[i][2:] += boxes_filt_0[i][:2]
                    boxes_filt_0 = boxes_filt_0.cpu()
                    
                    for i in range(boxes_filt_1.size(0)):
                        boxes_filt_1[i] = boxes_filt_1[i] * torch.Tensor([W, H, W, H])
                        boxes_filt_1[i][:2] -= boxes_filt_1[i][2:] / 2
                        boxes_filt_1[i][2:] += boxes_filt_1[i][:2]
                    boxes_filt_1 = boxes_filt_1.cpu()
                    
                    transformed_boxes_0 = predictor.transform.apply_boxes_torch(boxes_filt_0, image.shape[:2]).to(device)
                    transformed_boxes_1 = predictor.transform.apply_boxes_torch(boxes_filt_1, image.shape[:2]).to(device)

                    
                    if m0!=0 and m1!=0:
                        
                        masks_0, _, _ = predictor.predict_torch(  #masks_0[0]:[1,320,576]
                            point_coords = None,
                            point_labels = None,
                            boxes = transformed_boxes_0.to(device),
                            multimask_output = False,
                        )
                        plt.figure(figsize=(10, 10))
                        plt.imshow(image)
                        for mask in masks_0:
                            # show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
                            show_mask(mask.cpu().numpy(), plt.gca(), random_color=False)
                        cnt = 0
                        for box, label in zip(boxes_filt_0, pred_phrases_0):
                            label_1 = label+f"_{cnt}"
                            show_box(box.numpy(), plt.gca(), label_1)
                            cnt+=1
                        plt.axis('off')
                        os.makedirs(os.path.join(output_dir,videos[k],image_name.split('.')[0]),exist_ok=True)
                        plt.savefig(
                            os.path.join(output_dir,videos[k],image_name.split('.')[0],f"grounded_sam_output_{object_1}.jpg"),
                            bbox_inches="tight", dpi=300, pad_inches=0.0
                        )
                        
                        
                        masks_1, _, _ = predictor.predict_torch(
                            point_coords = None,
                            point_labels = None,
                            boxes = transformed_boxes_1.to(device),
                            multimask_output = False,
                        )
                        plt.figure(figsize=(10, 10))
                        plt.imshow(image)
                        for mask in masks_1:
                            # show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
                            show_mask(mask.cpu().numpy(), plt.gca(), random_color=False)
                        cnt = 0
                        for box, label in zip(boxes_filt_1, pred_phrases_1):
                            label_1 = label+f"_{cnt}"
                            show_box(box.numpy(), plt.gca(), label_1)
                            cnt+=1
                        plt.axis('off')
                        os.makedirs(os.path.join(output_dir,videos[k],image_name.split('.')[0]),exist_ok=True)
                        plt.savefig(
                            os.path.join(output_dir, videos[k],image_name.split('.')[0],f"grounded_sam_output_{object_2}.jpg"),
                            bbox_inches="tight", dpi=300, pad_inches=0.0
                        )
                    
                    
                        record_all_good_spatial = []
                        for ii in range(len(clean_boxes_0)):
                            for jj in range(len(clean_boxes_1)):  
                                intersection, IoU, IoMinA, XIoMinX, YIoMinY = intersection_judge(clean_boxes_0[ii],clean_boxes_1[jj])
                                if intersection!=0:
                                    depth_map = cv2.imread(depth_path,cv2.IMREAD_GRAYSCALE)
                                    height, width = depth_map.shape 
                                    mask_image_0 = (masks_0[ii].cpu().numpy().squeeze() * 255).astype(np.uint8)
                                    obj1_seg = cv2.bitwise_and(depth_map, depth_map, mask=mask_image_0) 
                                    
                                    d1 = np.sum(obj1_seg)/cv2.countNonZero(mask_image_0)
                                    mask_image_1 = (masks_1[jj].cpu().numpy().squeeze() * 255).astype(np.uint8)
                                    obj2_seg = cv2.bitwise_and(depth_map, depth_map, mask=mask_image_1)
                                    d2 = np.sum(obj2_seg)/cv2.countNonZero(mask_image_1)
                                    
                                    seg_save_path = os.path.join(output_dir, videos[k],image_name.split('.')[0],f"obj1_seg_{ii}.png")
                                    cv2.imwrite(seg_save_path, obj1_seg)
                                    seg_save_path = os.path.join(output_dir, videos[k],image_name.split('.')[0],f"obj2_seg_{jj}.png")
                                    cv2.imwrite(seg_save_path, obj2_seg)
                                    
                                    if (not 0 <= d1 <= 255) or (not 0 <= d2 <= 255) :
                                        print("d1 wrong value")
                                    if spatial == "in front of":
                                        if d1>d2:
                                            
                                            prob_score = 0.5*probs_0[ii]+0.5*probs_1[jj]
                                            spatial_score_1 = IoU
                                            total_score_1 = 0.5*prob_score + 0.5*spatial_score_1
                                            
                                            info = {}
                                            info["name"]=f"{ii}_{jj}"
                                            info["box0"]=clean_boxes_0[ii]
                                            info["box1"]=clean_boxes_1[jj]
                                            info["total_score_1"] = total_score_1
                                            info["spatial_score_1"] = spatial_score_1
                                
                                            record_all_good_spatial.append(info)
                                            
                                    elif spatial == "behind":
                                        if d1<d2:
                                            seg_save_path = os.path.join(output_dir, videos[k],image_name.split('.')[0],f"obj1_seg_{ii}.png")
                                            cv2.imwrite(seg_save_path, obj1_seg)
                                            seg_save_path = os.path.join(output_dir, videos[k],image_name.split('.')[0],f"obj2_seg_{jj}.png")
                                            cv2.imwrite(seg_save_path, obj2_seg)
                                            
                                            prob_score = 0.5*probs_0[ii]+0.5*probs_1[jj]
                                            spatial_score_1 = IoU
                                            total_score_1 = 0.5*prob_score + 0.5*spatial_score_1
                                            
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
                                        
                                
                    
                    csv_writer.writerow([videos[k],image_name, prompt, m0,m1, score_1])
                    csvfile.flush()   
    
    combine_frame(f'{args.csv_dir}/3d_frame.csv', f'{args.csv_dir}/3d_video.csv')     
            
            
            
            
        