# CDNMF: Contrastive Deep Nonnegative Matrix Factorization for Community Detection (ICASSP 2024)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/contrastive-deep-nonnegative-matrix/community-detection-on-citeseer)](https://paperswithcode.com/sota/community-detection-on-citeseer?p=contrastive-deep-nonnegative-matrix)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/contrastive-deep-nonnegative-matrix/community-detection-on-pubmed)](https://paperswithcode.com/sota/community-detection-on-pubmed?p=contrastive-deep-nonnegative-matrix)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/contrastive-deep-nonnegative-matrix/community-detection-on-cora)](https://paperswithcode.com/sota/community-detection-on-cora?p=contrastive-deep-nonnegative-matrix)


The implementation of our paper ["Contrastive Deep Nonnegative Matrix Factorization for Community Detection"](https://arxiv.org/abs/2311.02357). (**ICASSP 2024, CCF B**)

**Blog**: [[CN]](https://blog.csdn.net/m0_50826544/article/details/136773553?spm=1011.2415.3001.5331) | [[EN]](https://medium.com/@liych168168/cdnmf-a-trustworthy-and-efficient-community-detection-method-88dc9a6aba2b)

**Wechat Public Platform**: [Paper Reading Note](https://mp.weixin.qq.com/s/3rknZMBeqHW0fNtEvrEA8Q)

**Academic Poster in ICASSP2024 (Korea)**: [Poster](https://github.com/6lyc/CDNMF/blob/main/Poster.pdf)

## Overview
![framework](./figures/framework.png)
We introduce the idea of contrastive learning (CL) into the nonnegative matrix factorization (NMF) for community detection (CD) **for the first time**, solving the problems of the existing work.

The two modules are mutually reinforcing and naturally coupled: On the one hand, with the help of CL, positive samples are contrasted to allow the NMF to learn **both the graph topological and attribute signals**, while negative samples are contrasted to make the embedding space from NMF more suitable for CD; On the other hand, with the help of NMF's outputs, **more accurate negative samples** can be obtained in CL. Moreover, NMF increases the interpretability of the existing framework.

## Abstract
Recently, nonnegative matrix factorization (NMF) has been widely adopted for community detection, because of its better interpretability. However, the existing NMF-based methods have the following three problems: 1. they directly transform the original network into community membership space, so it is difficult for them to capture the hierarchical information; 2. they often only pay attention to the topology of the network and ignore its node attributes; 3. it is hard for them to learn the global structure information necessary for community detection. Therefore, we propose a new community detection algorithm, named Contrastive Deep Nonnegative Matrix Factorization (CDNMF). Firstly, we deepen NMF to strengthen its capacity for information extraction. Subsequently, inspired by contrastive learning, our algorithm creatively constructs network topology and node attributes as two contrasting views. Furthermore, we utilize a debiased negative sampling layer and learn node similarity at the community level, thereby enhancing the suitability of our model for community detection. We conduct experiments on three public real graph datasets and the proposed model has achieved better results than state-of-the-art methods.

## Requirements
To install the dependencies: `pip install -r requirements.txt`.

## DataSets
The code takes an input graph in a txt file. Sample graphs for the **Cora**, **Citeseer**, **PubMed** and **Wiki** is included in the `Database/` directory.

### Art Institute of Chicago

To crawl 25,000 unique artworks with artist IDs, titles, and artwork-type labels, then prepare up to 21,000 of them:

```powershell
python crawl_artic_cdnmf.py
```

To run the two phases in separate notebook cells, open `crawl_artic_cdnmf.ipynb` from the repository root and run the crawl cell followed by the preprocessing cell. The command-line equivalent is:

```powershell
python crawl_artic_cdnmf.py --stage crawl
python crawl_artic_cdnmf.py --stage preprocess
```

During crawling, the script scans API pages and saves unique artworks with a valid ID, `artist_id`, title, and non-empty `artwork_type_title`. It continues paging until it has up to 25,000 records or reaches the end of the API collection. The complete crawl is saved as `artworks_raw.csv`; progress is saved after every page in `artworks_raw.checkpoint.json`. If interrupted, run the crawl stage again with the same output path and target to resume from the next API page. Do not delete or edit only one of the CSV/checkpoint files while resuming.

Preprocessing uses the top 6–8 artwork types with at least 100 samples, and iteratively removes classes/artists that no longer satisfy the thresholds. The largest retained class is capped at 3 times the smallest class. An artist contributes at most 200 artworks, and each retained artist has at least 11 artworks. The default text features are `material_titles`, `technique_titles`, `place_of_origin`, and `medium_display`; whichever of `subject_titles` or `style_titles` has higher coverage is added if it reaches 50% coverage. `title` and `artist_title` are retained as metadata but are not input features. Unicode NFKC, HTML entity decoding, case-folding, HTML-tag/punctuation removal, and whitespace normalization are applied before TF-IDF. The fixed evaluation label is `artwork_type_title`, excluded from input features.

The graph connects works by the same artist, with no more than 20 neighbors per node. `walks.txt` contains the same author-based graph pairs for the model's smoothness term; it does not run expensive full-corpus random walks. CSV outputs include `artworks_raw.csv`, `artworks.csv`, `nodes.csv`, `edges.csv`, and `label_mapping.csv`; `metadata.json` summarizes the dataset, including retained class sizes and feature coverage. `nodes.csv` maps each `node_index` to the artwork ID, artist ID, title, artist title, and label. `edges.csv` stores endpoints of undirected same-artist links. CDNMF also needs `edges.txt`, `features.txt`, `group.txt`, and `walks.txt`, so these compatibility files remain in the output directory alongside CSVs. If the available data cannot meet the class/artist thresholds, preprocessing reports an error instead of silently producing an invalid dataset.

To prepare a smaller dataset from the saved crawl for the current model:

```powershell
python crawl_artic_cdnmf.py --stage preprocess --input-csv Database\artic_25000\artworks_raw.csv --output-dir Database\artic_2000 --processed-limit 2000
```

**Memory limitation:** the current CDNMF implementation materializes dense node-by-node matrices and runs full-batch NMF. The 21,000-node export is suitable as a prepared dataset, but the existing model is not practical to train on all of it; use a smaller prepared subset or first replace the model with a sparse/mini-batch implementation. The script does not claim a higher evaluation score; scores depend on the data and training setup.

### arXiv author-overlap graph

Run the crawler in a Kaggle notebook cell or a local terminal:

```python
!python script_arxiv.py --output-dir /kaggle/working/arxiv
```

The script queries the public arXiv Atom API for papers submitted from 2020 through 2024 in `cs.CV`, `hep-th`, `q-bio.NC`, `math.CO`, and `astro-ph.GA`. It uses those category searches as seeds, then performs bounded breadth-first expansion through matching author names. Defaults are 1,000 seed papers per category, at most 10,000 papers overall, BFS depth 2, at most 100 papers per author query, and at most 250 author queries. API requests are spaced by at least three seconds. Adjust these limits with command-line options; a full crawl may take a long time.

Only papers whose primary category is one of the requested categories and whose author count is at most 15 are kept. The final undirected graph joins papers with at least one normalized author name in common, with edge `weight` equal to the number of shared names. Isolated papers are removed, followed by all but the largest connected component. Author names are not unique identifiers, so normalization can incorrectly merge different people; this is an inherent limitation of the metadata.

The output directory contains `nodes.csv`, `edges.csv`, `label_mapping.csv`, and `arxiv_author_graph.graphml`. `nodes.csv` contains the primary-category label and arXiv paper URL; `edges.csv` gives paper endpoints, the shared-author count, and matching author names. For example, to change crawl limits:

```python
!python script_arxiv.py --seed-per-category 500 --max-papers 5000 --max-depth 1 --output-dir /kaggle/working/arxiv
```

## Quick Start
Running shell for Cora：    
`python script_cora.py`

Running shell for Citeseer：    
`python script_citeseer.py`

Running shell for PubMed：    
`python script_pubmed.py`

## Community Detection Results and Convergence
Performance on **Cora**, **Citeseer** and **PubMed** (20 runs):  
![results](./figures/results.png) 

Convergence Rate on **Cora**, **Citeseer**:  
![convergence](./figures/convergence.png)

## Citation
Please cite our paper if you use this code or our model in your own work:

```BibTex
@inproceedings{li2024contrastive,  
  title={Contrastive deep nonnegative matrix factorization for community detection},  
  author={Li, Yuecheng and Chen, Jialong and Chen, Chuan and Yang, Lei and Zheng, Zibin},  
  booktitle={ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},  
  pages={6725--6729},  
  year={2024},  
  organization={IEEE}  
}
```
