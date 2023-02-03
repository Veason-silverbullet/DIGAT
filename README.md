# DIGAT
This repository releases the code of paper [**DIGAT: Modeling News Recommendation with Dual-Graph Interaction** (EMNLP-2022 Findings)](https://aclanthology.org/2022.findings-emnlp.491.pdf).
<br/><br/>

## Dataset Preparation
The experiments are conducted on MIND-small and MIND-large. Our code tries to download and extract MIND-small and MIND-large to the directories `../MIND-small` and `../MIND-large`. Since MIND is quite large, if our code cannot download it due to network issues, please manually download MIND with the script `download_MIND.sh`.
<br/><br/>

## Environment Requirements
Dependencies are needed to be installed by
<pre><code>bash install_dependencies.sh</code></pre>
Our experiments require python>=3.7, torch==1.12.0, and torch_scatter==2.0.9. The [torch_scatter](https://github.com/rusty1s/pytorch_scatter) package is necessary. If the dependency installation fails, please follow [https://github.com/rusty1s/pytorch_scatter](https://github.com/rusty1s/pytorch_scatter) to install the package manually.
<br/><br/>

## Experiment Running
<hr>Training DIGAT
<pre><code>python main.py --graph_encoder=DIGAT</code></pre>
<pre><code># Use two GPUs to train DIGAT with DDP
python -m torch.distributed.launch --nproc_per_node=2 main.py --graph_encoder=DIGAT</code></pre>

<hr>Experiments in Section 4.4
<pre><code>python main.py --graph_encoder=wo_SA
python main.py --graph_encoder=Seq_SA</code></pre>

<hr>Experiments in Section 4.5
<pre><code>python main.py --graph_encoder=wo_interaction
python main.py --graph_encoder=news_graph_wo_inter
python main.py --graph_encoder=user_graph_wo_inter</code></pre>

<hr>Experiments in Section 4.6
<pre><code>python main.py --graph_encoder=DIGAT --SAG_neighbors=1 --SAG_hops=2
python main.py --graph_encoder=DIGAT --SAG_neighbors=2 --SAG_hops=2
python main.py --graph_encoder=DIGAT --SAG_neighbors=3 --SAG_hops=2
python main.py --graph_encoder=DIGAT --SAG_neighbors=4 --SAG_hops=2
python main.py --graph_encoder=DIGAT --SAG_neighbors=5 --SAG_hops=2
python main.py --graph_encoder=DIGAT --SAG_neighbors=6 --SAG_hops=2
python main.py --graph_encoder=DIGAT --SAG_neighbors=7 --SAG_hops=2
python main.py --graph_encoder=DIGAT --SAG_neighbors=8 --SAG_hops=2
python main.py --graph_encoder=DIGAT --SAG_neighbors=3 --SAG_hops=1
python main.py --graph_encoder=DIGAT --SAG_neighbors=3 --SAG_hops=3
python main.py --graph_encoder=DIGAT --SAG_neighbors=3 --SAG_hops=4</code></pre>

<hr>Experiments in Section 4.7
<pre><code>python main.py --graph_encoder=DIGAT --graph_depth=1
python main.py --graph_encoder=DIGAT --graph_depth=2
python main.py --graph_encoder=DIGAT --graph_depth=3
python main.py --graph_encoder=DIGAT --graph_depth=4
python main.py --graph_encoder=DIGAT --graph_depth=5
python main.py --graph_encoder=DIGAT --graph_depth=6
python main.py --graph_encoder=DIGAT --graph_depth=7</code></pre>
<br/>


## Experiments on MIND-small and MIND-large
The experiment dataset can be specified by the config parameter `--dataset=[MIND-small,MIND-large] (default MIND-small)`.
<pre><code>python main.py --dataset=MIND-small
python main.py --dataset=MIND-large</code></pre>
For MIND-large, please submit the model prediction file to [*MIND leaderboard*](https://msnews.github.io/index.html#leaderboard) for performance evaluation. For example, having finished training model #1, the model prediction file is at `prediction/MIND-large/MSA-DIGAT/#1/prediction.zip`. If the prediction zip file is not found, please find the raw prediction file at `test/MIND-large/res/MSA-DIGAT/best_model_MIND-large_MSA-DIGAT_#1_MSA-DIGAT/MSA-DIGAT.txt`.
<br/><br/>

## Faster Inference
We had benchmarked and found that the most computation overhead comes from Eq. (8) in the paper. Hence, we perform partial quantization in computing Eq. (8). For the faster inference code, please checkout to the branch `faster-inference`:
<pre><code>git checkout faster-inference</code></pre>
In `faster-inference` mode, the inference time on MIND-small reduces from around 600s to 400s (benchmarked on Nvidia 3090), where we observe no AUC/MRR/nDCG performance degradation (accurate to 1e-4).

It is also worth noting that 1) quantization is only partially performed in Eq. (8), the other parts of DIGAT are still computed in fp32. 2) Do NOT perform quantization in training DIGAT, which will degrade the performance. 3) Concretely, the computation overhead lies within the huge broadcast-add [K3 + K1 + K2](https://github.com/Veason-silverbullet/DIGAT/blob/6cfdaffae5d749bd12156084d27c08d0ba4011a6/graphEncoders.py#L150). This broadcast-add may be optimized by tailored efficient CUDA operator in the future.

## TODO features (may be updated in the future)
1. DIGAT + PLM news encoder
2. FP16 training

P.S. As the majority of this work is done in CUHK, we do not have enough GPUs to train PLM news encoder on MIND-large (due to millions of user logs). We hope to experiment DIGAT with PLM news encoder in the future. For anyone interested in training DIGAT with PLM news encoder, please feel free to contact us via zmmao@se.cuhk.edu.hk, and we are delighted to share the code.


## Citation
```
@inproceedings{mao-etal-2022-digat,
    title = "{DIGAT}: Modeling News Recommendation with Dual-Graph Interaction",
    author = "Mao, Zhiming  and
              Li, Jian  and
              Wang, Hongru  and
              Zeng, Xingshan  and
              Wong, Kam-Fai",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2022",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-emnlp.491",
    pages = "6595--6607"
}
```
The paper was first submitted to ARR 2021 November [[link](https://openreview.net/forum?id=t2vXlG7Oe5m)].
