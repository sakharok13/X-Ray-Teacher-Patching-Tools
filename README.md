# X-Ray Distillation: Object-Completion tools

[![arXiv](http://img.shields.io/badge/cs.CV-arXiv:2404.00679-B31B1B.svg)](https://arxiv.org/abs/2404.00679)

![Poster](./images/poster.png)

### Abstract

This paper addresses the critical challenges of sparsity and occlusion in LiDAR-based 3D object detection. Current methods often rely on supplementary modules or specific architectural designs, potentially limiting their applicability to new and evolving architectures. To our knowledge, we are the first to propose a versatile technique that seamlessly integrates into any existing framework for 3D Object Detection, marking the first instance of Weak-to-Strong generalization in 3D computer vision. We introduce a novel framework, X-Ray Distillation with Object-Complete Frames, suitable for both supervised and semi-supervised settings, that leverages the temporal aspect of point cloud sequences. This method extracts crucial information from both previous and subsequent LiDAR frames, creating Object-Complete frames that represent objects from multiple viewpoints, thus addressing occlusion and sparsity. Given the limitation of not being able to generate Object-Complete frames during online inference, we utilize Knowledge Distillation within a Teacher-Student framework. This technique encourages the strong Student model to emulate the behavior of the weaker Teacher, which processes simple and informative Object-Complete frames, effectively offering a comprehensive view of objects as if seen through X-ray vision. Our proposed methods surpass state-of-the-art in semi-supervised learning by 1-1.5 mAP and enhance the performance of five established supervised models by 1-2 mAP on standard autonomous driving datasets, even with default hyperparameters.

##
- [OpenPCDet repository is used for training SSL and supervised models](https://github.com/open-mmlab/OpenPCDet)
- [Official DSVT repository is used for training DSVT models](https://github.com/Haiyang-W/DSVT).

## Demos

All videos are hosted on YouTube platform.

- [Original vs. Patched Scenes](https://youtu.be/GN-Bn7nVqZc?si=xU9xGubApSAKLc1P)
- [X-Ray Teacher vs. Baseline Model](https://youtu.be/4l08cgaCSkg?si=Hjq3j1GWN9LQsFF9)
- [[CVPR 2024] Week-to-Strong 3D Object Detection With X-Ray Distillation](https://youtu.be/qo6Z3Bk_I9M?si=qqty2WDliksbwECW)

## Core Contributors

- [Alex Dadukin](https://github.com/st235)
- [Alexander Gambashidze](https://github.com/sakharok13)
- [Maria Razzhivina](https://github.com/mariarzv)

## Citations

```tex
@misc{gambashidze2024weaktostrong,
      title={Weak-to-Strong 3D Object Detection with X-Ray Distillation}, 
      author={Alexander Gambashidze and Aleksandr Dadukin and Maksim Golyadkin and Maria Razzhivina and Ilya Makarov},
      year={2024},
      eprint={2404.00679},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgements

- [Greedy Grid Search](https://github.com/DavidBoja/greedy-grid-search): the code is used for one of [the accumulation strategies](./src/accumulation/greedy_grid_accumulator_strategy.py). The forked code of Greedy Grid is located under [`src.utils.greedy_grid`](./src/utils/greedy_grid).

