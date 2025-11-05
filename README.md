## Hardware Specifications
This project was run on NVIDIA RTX 4090 GPU
## How to run
1.  **Clone the repository:**
2.  **Install dependencies:**
    The code was tested on Python 3.10 in Conda Environment. Run the following
    ```bash
    conda env create -f environment.yml
    conda activate unet
    ```
3. **Download the dataset**
    Note: As we can upload all the images into github, redownloading the dataset is unnecessary.
    
    The ISBI-2012 EM Segmentation Challenge dataset can be downloaded on https://downloads.imagej.net/ISBI-2012-challenge.zip
    We converted the tif images into individual images, then save all the directories into /data directory. The specific paths for all the directories (test, train, val) can be found and edited on config.py. In our case, we chose the 20th image as validation data.
4.  **Train on your own images**
    Put all your training image-ground truth pairs in the data directory, where the paths are specified in config.py
    Then, run
    ```bash
    python3 train.py
    ```
    Alternatively, you can download the pretrained model (on ISBI 2012 EM Segmentation Challenge) at https://drive.google.com/file/d/15AWBbmBZI-zksuFbL0cFkHmAygpwO70S/view?usp=sharing

    The training loss curve will be saved on /results/loss directory
5. **Test on your own images**
    Put all your testing image-ground truth pairs in the data directory, where the paths are specified in config.py
    Then, run
    ```bash
    python3 test.py
    ```
    In the results/images directory, you can find a directory that stores gt.tif and predictions.tf. Compare the 2 tif files using standard benchmark criteria by running test.bsh in Fiji Imagej. The documentations can be found on https://imagej.net/tutorials/segmentation-evaluation-metrics, and the Fiji software can be downloaded on https://imagej.net/software/fiji/downloads
