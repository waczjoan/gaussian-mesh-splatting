# GaMeS
Joanna Waczyńska*, Piotr Borycki*, Sławomir Tadeja, Jacek Tabor, Przemysław Spurek
(* indicates equal contribution)<br>

This repository contains the official authors implementation associated with the paper "GaMeS: Mesh-Based Adapting and Modification of Gaussian Splatting".

Abstract: *
Recently, a range of neural network-based methods for image rendering have been introduced. 
One such widely-researched neural radiance field (NeRF) relies 
on a neural network to represent 3D scenes, allowing for realistic view synthesis
from a small number of 2D images. However, most NeRF models are constrained by 
long training and inference times. 
In comparison, Gaussian Splatting (GS) is a novel, state-of-the-art technique
for rendering points in a 3D scene by approximating their contribution to image 
pixels through Gaussian distributions, warranting fast training and swift,
real-time rendering. A drawback of GS is the absence of a well-defined approach 
for its conditioning due to the necessity to condition several hundred thousand 
Gaussian components. To solve this, we introduce the Gaussian Mesh Splatting (GaMeS) model, 
which allows modification of Gaussian components in a similar way as meshes.
We parameterize each Gaussian component by the vertices of the mesh face. 
Furthermore, our model needs mesh initialization on input or estimated mesh during
training. We also define Gaussian splats solely based on their location on the mesh,
allowing for automatic adjustments in position, scale, and rotation during animation.
As a result, we obtain a real-time rendering of editable GS.*


Check us if you want to make a flying hotdog:

<img src="./assets/hotdog_fly.gif" width="250" height="250"/> </br>
</br>

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
<h3 class="title">GaMeS Gaussian Mesh Splatting</h3>
    <pre><code>@Article{waczynska2024games,
      author         = {Kerbl,Joanna Waczyńska and Piotr Borycki and Sławomir Tadeja and Jacek Tabor and Przemysław Spurek},
      title          = {GaMeS: Mesh-Based Adapting and Modification of Gaussian Splatting},
      year           = {2024},
      eprint         = {2402.01459},
      archivePrefix  = {arXiv},
      primaryClass   = {cs.CV},
}
}</code></pre>
    <h3 class="title">Gaussian Splatting</h3>
    <pre><code>@Article{kerbl3Dgaussians,
      author         = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
      title          = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
      journal        = {ACM Transactions on Graphics},
      number         = {4},
      volume         = {42},
      month          = {July},
      year           = {2023},
      url            = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}</code></pre>
  </div>
</section>

Multiple animation is possible like dancing ficus:</br>
</br>
<img src="./assets/ficus_dance.gif" width="250" height="250"/>
<img src="./assets/ficus_leaves.gif" width="250" height="250"/>
</br>

# Installation

Since, the software is based on original Gaussian Splatting repository, for details regarding requirements,
we kindly direct you to check  [3DGS](https://github.com/graphdeco-inria/gaussian-splatting). Here we present the most important information.

### Requirements

- Conda (recommended)
- CUDA-ready GPU with Compute Capability 7.0+
- CUDA toolkit 11 for PyTorch extensions (we used 11.8)

## Clone the Repository with submodules

```shell
# SSH
git clone git@github.com:waczjoan/gaussian-mesh-splatting.git --recursive
```
or
```shell
# HTTPS
git clone https://github.com/waczjoan/gaussian-mesh-splatting.git --recursive
```

### Environment

#### Local Setup

To install the required Python packages we used 3.7 and 3.8 python and conda v. 24.1.0
```shell
conda env create --file environment.yml
conda gaussian_splatting_mesh
```
Common issues:
- Are you sure you downloaded the repository with the --recursive flag?
- Please note that this process assumes that you have CUDA SDK **11** installed, not **12**. if you encounter a problem please refer to  [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) repository.

### Available options

Our solution proposes several model types, which you set using the `gs_type` flag.
You can run the following models in the repository:
<details>
<summary><span style="font-weight: bold;">Available models </span></summary>

  #### gs
  Use it if you want to run basic gaussian splatting
  #### gs_mesh 
  GaMeS model -- gaussians align exactly on the surface of the mesh. Note, the dataset requires a mesh. Use `num_splats` to set number of Gaussian per face.
  #### gs_flat
  Basic gaussian splicing which one scale value is epsilion, thus the resulting gaussians are flat. Model used to parameterize the `gs_points` GaMeS model.
  #### gs_flame
  GaMeS model -- GS allowing parameterization of the FLAME model. Note, the FLAME model is required. Download FLAME models and landmark embedings and place them inside games/flame_splatting/FLAME folder, as shown [here](https://github.com/soubhiksanyal/FLAME_PyTorch).
  #### gs_multi_mesh
  GaMeS model -- different version of `gs_mesh`, when more meshes are available. Gaussians align exactly on the surface of the meshes. Define them using `meshes` flag.

During render there is one more model available:
#### gs_points
  GaMeS model - used to parameterize the flat Gaussian splatting model.
</details>
<br>

<details>
<summary><span style="font-weight: bold;">Additional command Line Arguments for train.py (based on 3DGS repo) </span></summary>

  #### --source_path / -s
  Path to the source directory containing a COLMAP or Synthetic NeRF data set.
  #### --model_path / -m 
  Path where the trained model should be stored (```output/<random>``` by default).
  #### --images / -i
  Alternative subdirectory for COLMAP images (```images``` by default).
  #### --eval
  Add this flag to use a MipNeRF360-style training/test split for evaluation.
  #### --resolution / -r
  Specifies resolution of the loaded images before training. If provided ```1, 2, 4``` or ```8```, uses original, 1/2, 1/4 or 1/8 resolution, respectively. For all other values, rescales the width to the given number while maintaining image aspect. **If not set and input image width exceeds 1.6K pixels, inputs are automatically rescaled to this target.**
  #### --data_device
  Specifies where to put the source image data, ```cuda``` by default, recommended to use ```cpu``` if training on large/high-resolution dataset, will reduce VRAM consumption, but slightly slow down training. Thanks to [HrsPythonix](https://github.com/HrsPythonix).
  #### --white_background / -w
  Add this flag to use white background instead of black (default), e.g., for evaluation of NeRF Synthetic dataset.
  #### --sh_degree
  Order of spherical harmonics to be used (no larger than 3). ```3``` by default.
  #### --convert_SHs_python
  Flag to make pipeline compute forward and backward of SHs with PyTorch instead of ours.
  #### --convert_cov3D_python
  Flag to make pipeline compute forward and backward of the 3D covariance with PyTorch instead of ours.
  #### --debug
  Enables debug mode if you experience erros. If the rasterizer fails, a ```dump``` file is created that you may forward to us in an issue so we can take a look.
  #### --debug_from
  Debugging is **slow**. You may specify an iteration (starting from 0) after which the above debugging becomes active.
  #### --iterations
  Number of total iterations to train for, ```30_000``` by default.
  #### --ip
  IP to start GUI server on, ```127.0.0.1``` by default.
  #### --port 
  Port to use for GUI server, ```6009``` by default.
  #### --test_iterations
  Space-separated iterations at which the training script computes L1 and PSNR over test set, ```7000 30000``` by default.
  #### --save_iterations
  Space-separated iterations at which the training script saves the Gaussian model, ```7000 30000 <iterations>``` by default.
  #### --checkpoint_iterations
  Space-separated iterations at which to store a checkpoint for continuing later, saved in the model directory.
  #### --start_checkpoint
  Path to a saved checkpoint to continue training from.
  #### --quiet 
  Flag to omit any text written to standard out pipe. 
  #### --feature_lr
  Spherical harmonics features learning rate, ```0.0025``` by default.
  #### --opacity_lr
  Opacity learning rate, ```0.05``` by default.
  #### --scaling_lr
  Scaling learning rate, ```0.005``` by default.
  #### --rotation_lr
  Rotation learning rate, ```0.001``` by default.
  #### --position_lr_max_steps
  Number of steps (from 0) where position learning rate goes from ```initial``` to ```final```. ```30_000``` by default.
  #### --position_lr_init
  Initial 3D position learning rate, ```0.00016``` by default.
  #### --position_lr_final
  Final 3D position learning rate, ```0.0000016``` by default.
  #### --position_lr_delay_mult
  Position learning rate multiplier (cf. Plenoxels), ```0.01``` by default. 
  #### --densify_from_iter
  Iteration where densification starts, ```500``` by default. 
  #### --densify_until_iter
  Iteration where densification stops, ```15_000``` by default.
  #### --densify_grad_threshold
  Limit that decides if points should be densified based on 2D position gradient, ```0.0002``` by default.
  #### --densification_interval
  How frequently to densify, ```100``` (every 100 iterations) by default.
  #### --opacity_reset_interval
  How frequently to reset opacity, ```3_000``` by default. 
  #### --lambda_dssim
  Influence of SSIM on total loss from 0 to 1, ```0.2``` by default. 
  #### --percent_dense
  Percentage of scene extent (0--1) a point must exceed to be forcibly densified, ```0.01``` by default.

</details>
<br>

### Train 
1. Download dataset and put it `data/` directory.
  - We use the `NeRF Syntetic` dataset available under the [link](https://immortalco.github.io/NeuralEditor/), more precisely [here](https://drive.google.com/drive/folders/1JDdLGDruGNXWnM1eqY1FNL9PlStjaKWi) and meshes `blend_files`[here](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)
  - For `gs_flame` we used dataset available under the [link](https://github.com/WojtekZ4/NeRFlame/tree/main), more precisely [here](https://drive.google.com/drive/folders/1znso9vWtrkYqdMrZU1U0-X2pHJcpTXpe?usp=share_link)
  - `The MipNeRF360` scenes are hosted by the paper authors  under the [link](https://jonbarron.info/mipnerf360/).

2. What scenario do you want check?
To train a model in general use:
```shell
train.py --eval -s <path to data>  -m <path to output> --gs_type <model_type> # use -w, if you want white background
```

  - if you don't have mesh (or you don't want use it):
  ```shell
  train.py --eval -s /data/hotdog -m output/hotdog_flat --gs_type gs_flat -w
  ```

  - if you have mesh (in data/hotdog you should have `mesh.obj` file):
  ```shell
  train.py --eval -s /data/hotdog -m output/hotdog_gs_mesh --gs_type gs_mesh -w
  ```

### Evaluation
To eval a model in general use:
```shell
python render.py -m <path to output> --gs_type <model_type> # Generate renderings
python metrics.py -m <path to output> --gs_type <model_type> # Compute error metrics on renderings
```
  - if you don't have mesh (or you don't want use it):
  ```shell
  render.py -m output/hotdog_flat --gs_type gs_flat
  ```
or
  ```shell
  render.py -m output/hotdog_flat --gs_type gs_points
  ```
then to calculate metrics:
```shell
python metrics.py -m <path to output> --gs_type <model_type> # Compute error metrics on renderings
```
### Modification
  - if you don't have mesh (or you don't want use it), and in fact you use pseudo-mesh:
  ```shell
  render_points_time_animated.py -m output/hotdog_flat  --skip_train
  ```
#### Please note if you use Ubuntu 22.04
You will need to install a few dependencies before running the project setup.
```shell
# Dependencies
sudo apt install -y libglew-dev libassimp-dev libboost-all-dev libgtk-3-dev libopencv-dev libglfw3-dev libavdevice-dev libavcodec-dev libeigen3-dev libxxf86vm-dev libembree-dev
# Project setup
cd SIBR_viewers
cmake -Bbuild . -DCMAKE_BUILD_TYPE=Release # add -G Ninja to build faster
cmake --build build -j24 --target install
```