# T2V-CompBench
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


## Detection-based Evaluation (3D spatial relationships)
We use Depth Anything + GroundingDINO + SAM to evaluate 3D spatial relationships ("in front of" & "behind").
#### 1: Install Requirements

This Evaluation metric is based on the official repositories of Depth Anything and GroundingSAM. You can refer to [Depth Anything's GitHub repository](https://github.com/LiheYoung/Depth-Anything/tree/main) and [GroundingSAM's GitHub repository](https://github.com/IDEA-Research/GroundingDINO/tree/main) for specific environment dependencies and weights.

#### 2. Prepare Evaluation Videos

Please put all the videos of spatial relationships (both 2D and 3D) together as described in the section above.

#### 3. Run the Evaluation Codes

After obtaining the official Depth Anything code, place the following evaluation scripts in the `Depth-Anything/` directory:

- `run_depth.py`

After obtaining the official GroundingSAM code, place the following evaluation scripts in the `Depth-Anything/` directory:

- `spatial_relationship_3d.py`
  
## Tracking-based Evaluation
