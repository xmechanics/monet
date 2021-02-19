# monet

Dual Beam-shear DInC

## Setup Environment

Install Anaconda, or Miniconda.
Use `conda.yaml` file to create the conda env for this project.
Run following command under project root:
   
    conda env create -f conda.yaml

Then activate the conda env `monet`, to run scripts:

    conda activate monet


If the `conda.yaml` has been changed since you created the conda environment, you can use the yaml to update it

    conda update -f conda.yaml

## Playground

Create a folder for your study case in `workspace/data`.
Then put gradient data in `.h5` format in a sub-folder.

### `z`-field reconstruction

Each `.h5` file has two data sets `gradz_x` and `gradz_y`.

Clone `template.py` as your own script. Modify it to run your job.


    ffmpeg -framerate 23 -i img/microribbon-%04d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p microribbon.mp4

### centerline evolution

Each sub-folder contains `.h5` file at its root containing extracted centerline data. 
This h5 file contains a dataset normally named `gx` (gradient along the centerline).

It also has a `FullFieldData` sub-folder contains full field at each time step.

Clone `evolution_template.py` as your own script. Modify it to run your job.

