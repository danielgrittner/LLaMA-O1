# LLaMA-O1: Open Large Reasoning Model Frameworks For Training, Inference and Evaluation With PyTorch and HuggingFace
Large Reasoning Models powered by Monte Carlo Tree Search (MCTS), Self-Play Reinforcement Learning, PPO, AlphaGo Zero's dua policy paradigm and Large Language Models!

**DISCLAIM: This repository was for personal experimentation only and has no connection with my employer or any third-party organization or institution.**
![alt text](image-1.png)
## Call for Contributors！
## Documents
[Documents](https://github.com/SimpleBerry/LLaMA-O1/wiki)
[From AlphaGO Zero to RLHF...TBD]()
## Datasets

[OpenLongCoT Dataset](https://huggingface.co/datasets/qq8933/OpenLongCoT-Pretrain)
## Pretraining


TBD: Pretrain Code, recommend using LLaMaFactory for now.
## RLSP Training

### Recommend Base LongCoT Model for experiments

[Gemma2-2B-OpenLongCoT](https://huggingface.co/qq8933/OpenLongCoT-Base-Gemma2-2B)

### Install
Setup Environments,

```
pip install torch transformers accelerate peft datasets 
```
Pull codes,
```
git clone https://github.com/SimpleBerry/LLaMA-O1
cd LLaMA-O1
git pull
```

### Training
Run training,
```
# cd LLaMA-O1
python online_train.py
```
Or run with Accelerate,
```
accelerate config
accelerate launch online_train.py
```


## Inference 

## Evaluation

## Citation
Please Please cite me if this repo is helpful for you!🥰
```

@article{zhang2024llama,
  title={LLaMA-Berry: Pairwise Optimization for O1-like Olympiad-Level Mathematical Reasoning},
  author={Zhang, Di and Wu, Jianbo and Lei, Jingdi and Che, Tong and Li, Jiatong and Xie, Tong and Huang, Xiaoshui and Zhang, Shufei and Pavone, Marco and Li, Yuqiang and others},
  journal={arXiv preprint arXiv:2410.02884},
  year={2024}
}

@article{zhang2024accessing,
  title={Accessing GPT-4 level Mathematical Olympiad Solutions via Monte Carlo Tree Self-refine with LLaMa-3 8B},
  author={Zhang, Di and Li, Jiatong and Huang, Xiaoshui and Zhou, Dongzhan and Li, Yuqiang and Ouyang, Wanli},
  journal={arXiv preprint arXiv:2406.07394},
  year={2024}
}

```
## License
This Repository was distributed under the License of MIT.

PS: Please reserve author information and citations in re-developments.

## Contact
```
di.zhang@ustc.edu
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=SimpleBerry/LLaMA-O1&type=Date)](https://star-history.com/#SimpleBerry/LLaMA-O1&Date)
