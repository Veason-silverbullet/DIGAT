# DIGAT
This repository releases the code of paper [**DIGAT: Modeling News Recommendation with Dual-Graph Interaction** (EMNLP-2022 Findings)](https://arxiv.org/pdf/2210.05196.pdf).
<br/><br/>

## Dataset Preparation
The experiments are conducted on MIND-small and MIND-large. Our code tries to download and extract MIND-small and MIND-large to the directories `../MIND-small` and `../MIND-large`. Since MIND is quite large, if our code cannot download it due to network issues, please manually download MIND with the script `download_MIND.sh`.
<br/><br/>


## Environment Requirements
Dependencies are needed to be installed by
<pre><code>bash install_dependencies.sh</code></pre>
Our experiments require python>=3.7, torch==1.12.1, and torch_scatter==2.0.9. The [torch_scatter](https://github.com/rusty1s/pytorch_scatter) package is neccessary. If the dependency installation fails, please follow [https://github.com/rusty1s/pytorch_scatter](https://github.com/rusty1s/pytorch_scatter) to install the package manually.
<br/><br/>


## Experiment Running
<hr>Training DIGAT
<pre><code>python main.py --graph_encoder=DIGAT</code></pre>

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


## TODO features (may be updated soon)
1. Distributed training
2. Faster inference
3. DIGAT + PLM news encoder

P.S. As the majority of this work is done in CUHK, we do not have enough GPUs to train PLM news encoder on MIND-large (due to millions of user logs). We hope to experiment DIGAT with PLM news encoder in the future. For anyone interested in training DIGAT with PLM news encoder, please feel free to contact us via zmmao@se.cuhk.edu.hk, and we are happy to share the code.