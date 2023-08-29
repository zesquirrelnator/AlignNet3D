# AlignNet3D

> Solving the challenge of aligning 3D data with significant geometric variations.

Stay updated on my website: https:

## Description

Developed by The Lawrenceville School Junior, Georges Casassovici, who joined machine learning autonomously, AlignNet3D represents the initial phase of a grander project encompassing LiDAR data and machine learning. It specifically targets the challenge of aligning 3D datasets that exhibit significant geometric variations if generated using different procedures or captured using different sensors
, a problem highlighted in the research paper "Domain Adaptation on Point Clouds via Geometry-Aware Implicits" - Arxiv 2112.09343

## Getting Started

1. **Prerequisites**: 
    - Ensure you have Python (3.x recommended) installed.
    - [PyTorch](https://pytorch.org/get-started/locally/)
    - [Open3D](https://www.open3d.org/)
    - [PointNet](https://github.com/dengyingxu/pointnet)

2. **Installation**:
    ```bash
    git clone https://github.com/zesquirrelnator/AlignNet3D.git
    cd AlignNet3D
    pip install -r requirements.txt
    ```

3. **Usage**:
    The official set of scripts is meant for a set of test pairs while the backup scripts are meant for only a single pair to train on.

    - Train the model:
      ```bash
      python alignment_train.py
      ```
    - Evaluate the model
      ```bash
      python eval.py
      ```

## Contributing

We welcome contributions!

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- "Domain Adaptation on Point Clouds via Geometry-Aware Implicits" for the insights into 3D data alignment challenges.
- Pierre Pontevia for the help in troubleshooting through discussion

