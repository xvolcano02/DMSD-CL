# DMSD-CL
This repo holds codes of the paper: ''Debiasing Multimodal Sarcasm Detection with Contrastive Learning'' published on the conference AAAI 2024.

## SET UP
```
* 1 You can get the ViT pretrained model we used from "https://github.com/lukemelas/PyTorch-Pretrained-ViT"
* 2 You can get the RoBERTa-base pretrained language model from "https://huggingface.co/roberta-base"
* 3 You can get the multi-modal sarcasm detection dataset from "https://github.com/headacheboy/data-of-multimodal-sarcasm-detection"
* 4 You can get the ood data from "https://drive.google.com/drive/folders/1k6CoxqqcU6u9C0x2GFGROWj_OKpQo-MQ?usp=drive_link"
```

## Run

```sh
cd main
python main.py
```

## Citation

If you find this repo useful in your research works, please consider citing:

```
@article{jia2023debiasing,
  title={Debiasing Multimodal Sarcasm Detection with Contrastive Learning},
  author={Jia, Mengzhao and Xie, Can and Jing, Liqiang},
  journal={arXiv preprint arXiv:2312.10493},
  year={2023}
}
```
