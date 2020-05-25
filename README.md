# Recognizing the topological phase transition by Variational Autoregressive Networks

Cite this work as,

L. Wang, Y. Jiang, L. He, and K. Zhou, ArXiv:2005.04857 [Cond-Mat] (2020).


## Getting Started

The code requires Python >= 3.7 and PyTorch >= 1.2. You can configure on CPU machine and accelerate with a recent Nvidia GPU card.

Other requirements,

    numpy==1.16.4
    torch==1.1.0
    torchvision==0.3.0
    uncertainties==3.1.1

## Running the tests

Run a small size example,

    python3 main_xy.py --ham fm --lattice sqr --L 4 --beta 1 --net pixelcnn_xy --net_depth 3 --net_width 16 --bias --lr_schedule --beta_anneal 0.998 --clip_grad 1 --save_step 10 --visual_step 10 --save_sample --max_step 100 --cuda -1

## Authors

* **Lingxiao Wang** - *Construct codes and write the preprint paper* - [Homepage](https://sites.google.com/view/lingxiao)
* **Yin Jiang** - *Check codes and provide physics guidance*
* **Lianyi He** - *Provide physics guidance and polish the article*
* **Kai Zhou** - *Lead the project and complete the article.*

## License

This project is licensed under the MIT License - see the LICENSE file for details
