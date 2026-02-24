# mmspao

Multi-modal spatial omics integration with views and combinations.

Experiments were executed on NVIDIA A40 of 46068MiB memory in linux with torch==2.1.0+cu121

## Overview

mmspao is an innovative algorithm for multimodal spatial omics analysis, its core advantage is that it can seamlessly integrate transcriptome, proteome, epigenetics, metabolome and other data types. Through the unique modal interactive feature extraction and adaptive fusion mechanism, it can significantly improve the accuracy and biological interpretability of spatial domain recognition. The algorithm uses Kolmogorov-Arnold Network (KAN) to generate mode specific embedding, and constructs interactive features by calculating the outer product between modes. Finally, combined with Leiden clustering, it realizes the collaborative analysis of multimodal data.

## install mmspao

```python
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install mmspao
# pip install numpy==1.26.4
```