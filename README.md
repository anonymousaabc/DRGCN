This is a implementation of DRGAT.

# Step 1: Git clone this repo.


```shell
git clone https://github.com/anonymousaabc/DRGCN.git

```


# Step 2: Install DGL environment.


```shell
cd DRGCN

sh drgat_env_install.sh
```


# Step 3: Experimental datasets

Two datasets are required to run this code. We have already upload datasets in the directory. You can skip this step.

One is ogbn-arxiv origin data, the directory is `./drgat/dataset/ogbn_arxiv/`. 

The other is ogbn-arxiv pretrained node features from GIANT-XRT, the directory is `./drgat/dataset/ogbn-arxiv-pretrain/`.


# Step 4: Run the experiment.

**GIANT+XRT+DRGAT:** Run runexp_drgat_ogbnarxiv.sh for reproducing our results for ogbn-arxiv dataset with GIANT-XRT features.

```shell
cd drgat

sh runexp_drgat_ogbnarxiv.sh

```

**GIANT+XRT+DRGAT+KD:** Run runexp_drgat_ogbnarxiv_kd.sh for reproducing our results for ogbn-arxiv dataset with GIANT-XRT features and KD.

```shell
cd drgat

sh runexp_drgat_ogbnarxiv_kd.sh

```


# Results

If execute correctly, you should have the following performance (using pretrained GIANT-XRT features).

Metrics | GIANT-XRT+DRGAT	| GIANT-XRT+DRGAT+KD
-------- | ----- | -----
Average val accuracy (%) |	77.16 ± 0.08 |	77.25 ± 0.06
Average test accuracy (%) |	76.11 ± 0.09 |	76.33 ± 0.08

Number of params: 2685527

Our hardware used for the experiments is Tesla P100-PCIE-16GB.

Remark: We do not carefully fine-tune DRGAT for our GIANT-XRT. It is possible to achieve higher performance by fine-tune it more carefully.



