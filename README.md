# T2V-CompBench
## MLLM-based Evaluation
We use LLaVA as the MLLM model to evaluate the four categories: consistent attribute binding, dynamic attribute binding, action and interaction
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

## Detection-based Evaluation

## Tracking-based Evaluation
