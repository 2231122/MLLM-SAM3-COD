# MLLM-SAM3-COD

Training-free camouflaged object detection with Qwen2.5-VL and SAM3. The pipeline follows four parts of the paper: Fine-grained Category Query (FCQ), SAM3 segmentation, Semantic-Geometric Dual Confirmation (SGDC), and Semantic-Geometric Reasoning Injection (SGRI).

## Preparation

Install Qwen2.5-VL according to the [official Qwen documentation](https://github.com/QwenLM/Qwen2.5-VL), then install SAM3 according to its official release. Stage 1 uses the Qwen environment; Stages 2–4 use the SAM3 environment. This repository includes the SAM3 source under `sam3-main`; place model checkpoints outside the repository or pass their paths with the command-line options.

```bash
pip install -r requirements.txt
```

## Pipeline

```bash
python Stage_1.py --image-dir data/images --output-dir outputs/fcq --model-path /path/to/Qwen2.5-VL
python Stage_2.py --image-dir data/images --annotation-dir outputs/fcq --output-dir outputs/sam3 --checkpoint /path/to/sam3.pt
python Stage_3.py --image-dir data/images --annotation-dir outputs/fcq --sam3-dir outputs/sam3 --output-dir outputs/sgdc
python Stage_4.py --image-dir data/images --sam3-dir outputs/sam3 --sgdc-dir outputs/sgdc --output-dir outputs/sgri
```

`Stage_1` produces FCQ annotations. `Stage_2` generates foreground and background masks. `Stage_3` confirms reliable masks and identifies hard samples. `Stage_4` forms grouped masks and geometric clues for the final SGRI reasoning step. `Ref-` retains the original experimental scripts for reference only.
