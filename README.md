# T2V-CompBench
## T2V-CompBench dataset
The T2V-CompBench dataset includes 700 prompts from 7 categories, each with 100 prompts. 

Each category is saved in a text file in the ```prompts/``` directory.


## MLLM-based Evaluation
We use LLaVA as the MLLM model to evaluate the four categories: consistent attribute binding, dynamic attribute binding, action binding and object interactions.
#### 1: Install Requirements

MLLM-based evaluation metrics are based on the official repository of LLaVA. You can refer to [LLaVA's GitHub repository](https://github.com/haotian-liu/LLaVA) for specific environment dependencies and weights.

#### 2. Prepare Evaluation Videos

Generate videos of your model using the T2V-CompBench prompts provided in the `prompts` directory. Organize them in the following structure (using the *dynamic attribute binding* category as an example):

```
../video/dynamic_attr
├── 0001.mp4
├── 0002.mp4
├── 0003.mp4
├── 0004.mp4
...
└── 0100.mp4
```

Note: The numerical names of the video files are just to indicate the reading order that matches the order of prompts. You can use other naming conventions that maintain the order (*e.g.* "0.mp4", "1.mpg", *etc.*)

#### 3. Run the Evaluation Codes

After obtaining the official LLaVA code, place the following evaluation scripts in the `LLaVA/llava/eval` directory:

- `consistent_attribute.py`
- `dynamic_attribute.py`
- `action_binding.py`
- `interaction.py`

Prepare the video repository path (*e.g.*, "../video/dynamic_attr") or a specific video path (*e.g.*, "../video/dynamic_attr/0001.mp4") in the argument `--video_grid_folder_prefix`. Configure the output path with the `--output-path` argument, and the file of prompts or metadata with the `--read-prompt-file` argument. The evaluation codes will automatically convert the video into three different formats (a standard video with 16 frames, frames, and an image grid) and then calculate the score.

##### Consistent Attribute Binding

Prepare the video path for `--video_grid_folder_prefix` and run the following command:

```
python llava/eval/consistent_attribute.py --video_grid_folder_prefix ../video/consistent_attribute
```

The output will be a CSV file named "consistent_attr_score.csv" in the "../csv_output_consistent_attr" directory. The video name, prompt, and score for each text-video pair will be recorded in the columns named of "name","prompt", "Score".

##### Dynamic Attribute Binding

Prepare the video path for `--video_grid_folder_prefix` and run the following command:

```
python llava/eval/dynamic_attribute.py --video_grid_folder_prefix ../video/dynamic_attribute
```

The output will be a CSV file named "dynamic_attr_score.csv" in the "../csv_output_dynamic_attr" directory. The video name, prompt, and score for each text-video pair will be recorded in the columns named of "name","prompt", "Score".

##### Action Binding

Input the grid format path and run:

```
python llava/eval/action_binding.py --video_grid_folder_prefix ../video/action_binding
```

The output will be a CSV file named "action_binding_score.csv" in the "../csv_output_action_binding" directory. The video name, prompt, and score for each text-video pair will be recorded in the columns named of "name","prompt", "Score".

##### Object Interaction

Input the grid format path and run:

```
python llava/eval/interaction.py --video_grid_folder_prefix ../video/interaction
```

The output will be a CSV file named "interaction.csv" in the "../csv_output_interaction" directory. The video name, prompt, and score for each text-video pair will be recorded in the columns named of "name","prompt", "Score".

**Evaluate Your Own Videos**

To evaluate your own videos, prepare the evaluation videos and prompt or metadata files similar to the provided examples. Follow the same steps to run the evaluation codes.

## Detection-based Evaluation (2D spatial relationships and generative numeracy)
We use GroundingDINO as the detection tool to evaluate the two categories: 2D spatial relationships and generative numeracy.
#### 1: Install Requirements

Detection-based Evaluation metrics are based on the official repository of GroundingDINO. You can refer to [GroundingDINO's GitHub repository](https://github.com/IDEA-Research/GroundingDINO/tree/main) for specific environment dependencies and weights.

#### 2. Prepare Evaluation Videos

Generate videos of your model using the T2V-CompBench prompts provided in the `prompts` directory. Organize them in the following structure (using the *spatial relationships* category as an example):

```
../video/spatial_relationships
├── 0001.mp4
├── 0002.mp4
├── 0003.mp4
├── 0004.mp4
...
└── 0100.mp4
```

Note: Please put all the videos of spatial relationships (both 2D and 3D) together. The numerical names of the video files are just to indicate the reading order that matches the order of prompts. You can use other naming conventions that maintain the order (*e.g.* "0.mp4", "1.mpg", *etc.*)

#### 3. Run the Evaluation Codes

After obtaining the official GroundingDINO code, place the following evaluation scripts in the `GroundingDINO/demo` directory:

- `spatial_relationship_2d.py`
- `generative_numeracy.py`

Compute the evaluation metric:

2D spatial relationship: 
```
python GroundingDINO/demo/spatial_relationship_2d.py
```

Generative numeracy:  
```
python GroundingDINO/demo/generative_numeracy.py
```

## Detection-based Evaluation (3D spatial relationships)
We use Depth Anything + GroundingSAM to evaluate 3D spatial relationships ("in front of" & "behind").
#### 1: Install Requirements

This Evaluation metric is based on the official repositories of Depth Anything and GroundingSAM. You can refer to [Depth Anything's GitHub repository](https://github.com/LiheYoung/Depth-Anything/tree/main) and [GroundingSAM's GitHub repository](https://github.com/IDEA-Research/GroundingDINO/tree/main) for specific environment dependencies and weights.

#### 2. Prepare Evaluation Videos

Please put all the videos of spatial relationships (both 2D and 3D) together as described in the section above.

#### 3. Run the Evaluation Codes

After obtaining the official Depth Anything code, place the following evaluation scripts in the `Depth-Anything/` directory:

- `run_depth.py`

The depth images will be stored in the `output_depth/` directory.

After obtaining the official GroundingSAM code, place the following evaluation scripts in the `Grounded-Segment-Anything/` directory:

- `spatial_relationship_3d.py`

Compute the evaluation metric:

step 1: 
```
python Depth-Anything/run_depth.py
```

step 2: 
```
python Grounded-Segment-Anything/spatial_relationship_3d.py
```
  
## Tracking-based Evaluation
We use GroundingSAM + DOT to evaluate motion binding.
#### 1: Install Requirements

This Evaluation metric is based on the official repositories of GroundingSAM and Dense Optical Tracking. You can refer to [GroundingSAM's GitHub repository](https://github.com/IDEA-Research/GroundingDINO/tree/main) and [Dense Optical Tracking's GitHub repository](https://github.com/16lemoing/dot?tab=readme-ov-file) for specific environment dependencies and weights.

#### 2. Prepare Evaluation Videos

Generate videos of your model using the T2V-CompBench prompts provided in the `prompts` directory. Organize them in the following structure (using the *spatial relationships* category as an example):

```
../video/motion_binding
├── 0001.mp4
├── 0002.mp4
├── 0003.mp4
├── 0004.mp4
...
└── 0100.mp4
```

Note: The numerical names of the video files are just to indicate the reading order that matches the order of prompts. You can use other naming conventions that maintain the order (*e.g.* "0.mp4", "1.mpg", *etc.*)

#### 3. Run the Evaluation Codes

After obtaining the official GroundingSAM code, place the following evaluation scripts in the `Grounded-Segment-Anything/` directory:

- `motion_binding_seg.py`

The background and forground segmentation of the 1st frame of the videos will be stored in the `output_motion_binding/` directory.

After obtaining the official DOT code, place the following evaluation scripts in the `dot/` directory:

- `motion_binding_foreground.py`
- `motion_binding_background.py`
- `process.py`

Then, replace the original`dot/dot/utils/options/demo_options.py` by 

- `demo_options.py`

Compute the evaluation metric:

step 1: 
```
python Grounded-Segment-Anything/motion_binding_seg.py
```

step 2: 
```
python dot/python motion_binding_foreground.py
```

step 3: 
```
python dot/python motion_binding_background.py
```

step 4: 
```
python dot/python process.py
```
