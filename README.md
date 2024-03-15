# GaMeS
Joanna Waczyńska*, Piotr Borycki*, Sławomir Tadeja, Jacek Tabor, Przemysław Spurek
(* indicates equal contribution)<br>

This repository contains the official authors implementation associated 
with the paper ["GaMeS: Mesh-Based Adapting and Modification of Gaussian Splatting"](https://arxiv.org/abs/2402.01459).

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
      author         = {Joanna Waczyńska and Piotr Borycki and Sławomir Tadeja and Jacek Tabor and Przemysław Spurek},
      title          = {GaMeS: Mesh-Based Adapting and Modification of Gaussian Splatting},
      year           = {2024},
      eprint         = {2402.01459},
      archivePrefix  = {arXiv},
      primaryClass   = {cs.CV},
}
</code></pre>
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
  Basic gaussian splicing which one scale value is epsilion, thus the resulting gaussians are flat. Model used to parameterize by the `gs_points` GaMeS model.
  #### gs_flame
  GaMeS model -- GS allowing parameterization of the FLAME model. Note, the FLAME model is required. Download FLAME models and landmark embedings and place them inside games/flame_splatting/FLAME folder, as shown [here](https://github.com/soubhiksanyal/FLAME_PyTorch).
  #### gs_multi_mesh
  GaMeS model -- different version of `gs_mesh`, when more meshes are available. Gaussians align exactly on the surface of the meshes. Define them using `meshes` flag.

During render there is one more model available:
#### gs_points
  GaMeS model - used to parameterize the flat Gaussian splatting model. Can not be used for training.
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

**## Quick start
In this section we describe general information; please find below section `Tutorial` for more details, or if you are here first time :)
### Train 
1. Download dataset and put it in `data` directory.
  - We use the `NeRF Synthetic`; dataset available under the [link](https://immortalco.github.io/NeuralEditor/), more precisely [here](https://drive.google.com/drive/folders/1JDdLGDruGNXWnM1eqY1FNL9PlStjaKWi) and meshes `blend_files`[here](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)
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

- for FLAME initiation mesh::
```shell
  train.py --eval -s /data/<id_face> -m output/<id_face> --gs_type gs_flame -w
  ```

### Evaluation
To eval a model in general use:
```shell
python scripts/render.py -m <path to output> --gs_type <model_type> # Generate renderings
python metrics.py -m <path to output> --gs_type <model_type> # Compute error metrics on renderings
```
Tip: If you have trouble with imports running `scrips/render.py` You should remember
```export PYTHONPATH=/path/to/a/project```

  - if you don't have mesh (or you don't want use it):
  ```shell
  scripts/render.py -m output/hotdog_flat --gs_type gs_flat
  ```
or
  ```shell
  scripts/render.py -m output/hotdog_flat --gs_type gs_points
  ```
then to calculate metrics:
```shell
python metrics.py -m <path to output> --gs_type <model_type> # Compute error metrics on renderings
```
### Modification
  - if you don't have mesh (or you don't want use it), and in fact you use pseudo-mesh:
  ```shell
  scripts/render_points_time_animated.py -m output/hotdog_flat  --skip_train
  ```

## Tutorial 
In this section we describe more details, and make step by step how to run GaMeS.

### Scenario I: we have mesh; and we want use it.
#### Dataset
1. Go to [nerf_synthetic](https://drive.google.com/drive/folders/1JDdLGDruGNXWnM1eqY1FNL9PlStjaKWi), download `hotdog` dataset and put it in to `data` directory. For example:

```
<gaussian-mesh-splatting>
|---data
|   |---<hotdog>
|   |---<ship>
|   |---...
|---train.py
|---metrics.py
|---...
```

2. Go to [blend_files](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) and download blender files. 
3. Open blender app; you need download it (https://www.blender.org/); open `hotdog.blend`. And save hotdog mesh: File -> Export -> Wavefront (.obj). File `mesh.obj` has to be in the same dir as dataset:
```
<gaussian-mesh-splatting>
|---data
|   |---<hotdog>
|   |   |---transforms_train.json
|   |   |---mesh.obj
|---train.py
|---metrics.py
|---...
```
4. Train model:
Using `rtx2070` it should take less than 15 minutes.
  ```shell
  train.py --eval -s /data/hotdog -m output/hotdog_gs_mesh --gs_type gs_mesh
  ```
Tip: In default, 2 Gaussians per face in mesh is used, to change it use `num_splats`. In fact, we highly recommend do it (in paper we used 5 or 10, check appendix), since it improves results, but training will take a bit longer, and in this tutorial, we would like it make it as easy it will be possible.
  ```shell
  train.py --eval -s /data/hotdog -m output/hotdog_gs_mesh --gs_type gs_mesh --num_splats 5 -w
  ```
Tip2: If you would like to, you can manually subdivide bigger faces in blender app.

In `output/hotdog_gs_mesh` you should find: 
```
<gaussian-mesh-splatting>
|---data
|   |---<hotdog>
|   |   |---transforms_train.json
|   |   |---mesh.obj
|   |   |---...
|---output
|   |---<hotdog_gs_mesh>
|   |   |---point_cloud
|   |   |---xyz
|   |   |---cfg_args
|   |   |---...
|---train.py
|---metrics.py
|---...
```
During training you should get information:
`Found transforms_train.json file, assuming Blender_Mesh data set!`

5. Evaluation:

Firstly let's check if our model correctly render files in init position. It should take less than 30 sec.

Tip: If you have trouble with imports running `scrips/render.py` You should remember
```export PYTHONPATH=/path/to/a/project```

In this scenario let's run:
  ```shell
  scripts/render.py -m output/hotdog_gs_mesh --gs_type gs_mesh
  ```
Use `--skip_train`, if you would like to skip train dataset in render.

Then, let's calculate  metrics (it takes around 3 minutes):
```shell
python metrics.py -m output/hotdog_gs_mesh --gs_type gs_mesh
```
In `output/hotdog_gs_mesh` you should find: 
```
<gaussian-mesh-splatting>
|---output
|   |---<hotdog_gs_mesh>
|   |   |---point_cloud
|   |   |---cfg_args
|   |   |---test
|   |   |---<ours_iter>
|   |   |   |---renders_gs_mesh
|   |   |---results_gs_mesh.json
|   |   |---...
|---metrics.py
|---...
```
In fact since it is just init position, you can use `gs` flag to render, and gets the same results.

7. Flying Hotdog:

Simply run:
```shell
  scripts/render_time_animated.py -m output/hotdog_gs_mesh # --skip_train
```
Please find renders in `time_animated` directory: 
```
<gaussian-mesh-splatting>
|---output
|   |---<hotdog_gs_mesh>
|   |   |---point_cloud
|   |   |---cfg_args
|   |   |---test
|   |   |---<ours_iter>
|   |   |   |---renders_gs_mesh
|   |   |   |---time_animated
|   |   |---...
|---metrics.py
|---...
```

If you want more transformation, we recommend you check `scripts/render_time_animated.py` file. Transformation `transform_hotdog_fly` is default, but there is a few more. You can also create your own modification.
7. Own  modification* (for blender users):

You can prepare your own more realistic transformation, for example an excavator lifting a shovel or spreading ficus branches, and save created mesh for example as `ficus_animate.obj`. 
Then you can use `render_from_mesh_to_mesh.py` file. 

### Scenario II: we don't have mesh; or we don't want use it.
#### Dataset
1. Go to [nerf_synthetic](https://drive.google.com/drive/folders/1JDdLGDruGNXWnM1eqY1FNL9PlStjaKWi), download `hotdog` dataset and put it in to `data` directory. For example:

```
<gaussian-mesh-splatting>
|---data
|   |---<hotdog>
|   |---<ship>
|   |---...
|---train.py
|---metrics.py
|---...
```

Here you don't need `mesh.obj`. If you already download it, it is okey, it can be in directory, it will be just not used.

2. Train Flat Gaussian Splatting.

First step is train simple flat Gaussian Splatting, use `gs_flat` flag. It should take around 10 minutes (using rtx2070).
  ```shell
  train.py --eval -s /data/hotdog -m output/hotdog_gs_flat --gs_type gs_flat -w
  ```

In `output/hotdog_flat` you should find: 
```
<gaussian-mesh-splatting>
|---data
|   |---<hotdog>
|   |   |---transforms_train.json
|   |   |---mesh.obj
|   |   |---...
|---output
|   |---<hotdog_gs_flat>
|   |   |---point_cloud
|   |   |---xyz
|   |   |---cfg_args
|   |   |---...
|---train.py
|---metrics.py
|---...
```
During training you should get information:
`Found transforms_train.json file, assuming Blender data set!`

4. Evaluation:

Firstly let's check you we can render Flat Gaussian Splatting:
```shell
  scripts/render.py -m output/hotdog_gs_flat --gs_type gs_flat
  ```
Use `--skip_train`, if you would like to skip train dataset in render.

Then, let's calculate  metrics (it takes around 3 minutes):
```shell
python metrics.py -m output/hotdog_gs_flat --gs_type gs_flat
```
In `output/hotdog_gs_flat` you should find: 
```
<gaussian-mesh-splatting>
|---output
|   |---<hotdog_gs_mesh>
|   |   |---point_cloud
|   |   |---cfg_args
|   |   |---test
|   |   |---<ours_iter>
|   |   |   |---renders_gs_flat
|   |   |---results_gs_flat.json
|   |   |---...
|---metrics.py
|---...
```

Since we would like to use parametrized Gaussians Splatting let's check renders after parametrization, use `gs_points` flag:
```shell
  scripts/render.py -m output/hotdog_gs_flat --gs_type gs_points #--skip_train
```

Then, let's calculate  metrics (it takes around 3 minutes):
```shell
python metrics.py -m output/hotdog_gs_flat --gs_type gs_points
```
In `output/hotdog_gs_flat` you should find: 
```
<gaussian-mesh-splatting>
|---output
|   |---<hotdog_gs_mesh>
|   |   |---point_cloud
|   |   |---cfg_args
|   |   |---test
|   |   |---<ours_iter>
|   |   |   |---renders_gs_flat
|   |   |   |---renders_gs_points
|   |   |---results_gs_flat.json
|   |   |---results_gs_points.json
|   |   |---...
|---metrics.py
|---...
```
Please note, `results_gs_flat` and `results_gs_points` are differ slightly, this is due to numerical calculations.

5. Modification / Wavy hotdog:

Simply run:
```shell
  scripts/render_points_time_animated.py -m output/hotdog_flat # --skip_train
```
Please find renders in `time_animated` directory: 
```
<gaussian-mesh-splatting>
|---output
|   |---<hotdog_gs_mesh>
|   |   |---point_cloud
|   |   |---cfg_args
|   |   |---test
|   |   |---<ours_iter>
|   |   |   |---renders_gs_flat
|   |   |   |---renders_gs_points
|   |   |   |---time_animated_gs_points
|   |   |---...
|---metrics.py
|---...
```

### Scenario III: we have initial mesh -- FLAME.

1. Go to  [NeRFlame](https://github.com/WojtekZ4/NeRFlame/tree/main), more precisely [here](https://drive.google.com/drive/folders/1znso9vWtrkYqdMrZU1U0-X2pHJcpTXpe?usp=share_link)
 and download `face_f1036_A` dataset and put it in to `data` directory. For example:

```
<gaussian-mesh-splatting>
|---data
|   |---<face_f1036_A>
|   |---...
|---train.py
|---metrics.py
|---...
```

Here you don't need `mesh.obj`. But... we use initial FLAME model. Hence:
Download FLAME model from official [website](https://flame.is.tue.mpg.de/). You need to sign up and agree to the model license for access to the model. Copy the downloaded models and put it in `games\flame_splatting\FLAME\model` folder (for more details see `games\flame_splatting\FLAME\config.py` file).

3. Train Flame Gaussian Splatting.

Train models with `gs_flame` flag. It should take around 50 minutes (using rtx2070).
  ```shell
  train.py --eval -s data/face_f1036_A -m output/face_f1036_A --gs_type gs_flame -w
  ```

In `output/face_f1036_A` you should find: 
```
<gaussian-mesh-splatting>
|---data
|   |---<face_f1036_A>
|   |   |---transforms_train.json
|   |   |---...
|---output
|   |---<face_f1036_A>
|   |   |---point_cloud
|   |   |---xyz
|   |   |---cfg_args
|   |   |---...
|---train.py
|---metrics.py
|---...
```
During training you should get information:
"Found transforms_train.json file, assuming Flame Blender data set!"

4. Evaluation:

Firstly let's check you we can render original Gaussian Splatting (since we save scaling, ration etc you can use `gs` flag):
```shell
  scripts/render.py -m output/face_f1036_A --gs_type gs
  ```
Use `--skip_train`, if you would like to skip train dataset in render. You should see "assuming Blender data set" information.

Then, let's calculate  metrics (it takes around 2 minutes):
```shell
python metrics.py -m output/face_f1036_A --gs_type gs
```
In `output/face_f1036_A` you should find: 
```
<gaussian-mesh-splatting>
|---output
|   |---<face_f1036_A>
|   |   |---point_cloud
|   |   |---cfg_args
|   |   |---test
|   |   |---<ours_iter>
|   |   |   |---renders_gs
|   |   |---results_gs.json
|   |   |---...
|---metrics.py
|---...
```

Since we would like to use parametrized Flame Gaussians Splatting let's check renders after parametrization, use `gs_flame` flag:
```shell
  scripts/render_flame.py -m output/face_f1036_A #--skip_train
```
Please note, you will see "assuming Flame Blender data set" information.

In `output/face_f1036_A` you should find: 
```
<gaussian-mesh-splatting>
|---output
|   |---<face_f1036_A>
|   |   |---point_cloud
|   |   |---cfg_args
|   |   |---test
|   |   |---<ours_iter>
|   |   |   |---renders_gs_flame
|   |   |---...
|---metrics.py
|---...
```

Renders `renders_gs_flame` and `renders_gs` should correspond to each other, that is, give the same results (except numerical differences). 

5. Modification:
If you would like to change expression or position or any FLAME parameter please check `render_set_animated` function in `scripts\redner_flame.py` -- you should manage how to animate! :))

For render use `animated` flag:
```shell
  scripts/render_flame.py -m output/face_f1036_A --animated #--skip_train
```

In `output/face_f1036_A` you should find: 
```
<gaussian-mesh-splatting>
|---output
|   |---<face_f1036_A>
|   |   |---point_cloud
|   |   |---cfg_args
|   |   |---test
|   |   |---<ours_iter>
|   |   |   |---flame_animated
|   |   |---...
|---metrics.py
|---...
```


###
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
Common problem:

1. We notice some of the people have problem "There are no g++ version bounds defined for CUDA", please check if gcc/g++ version is correct: [link](https://gist.github.com/ax3l/9489132); probably the downgrade will help: [link](https://webhostinggeeks.com/howto/how-to-downgrade-gcc-version-on-ubuntu/).
Tip: Firstly, you should find where is CUDA installed, check path `which g++`. 
