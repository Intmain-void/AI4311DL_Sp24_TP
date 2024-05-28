# DL_Sp24_TermProject_final

GIST spring semester Deep learning term project

---

## How to run `mosaicv1`

1. Create an environment with conda with python version 3.8
  ``` {powershell}
  conda create -n DLTerm python=3.8
  ```

2. Activate new environment
  ``` {powershell}
  conda activate DLTerm
  ```
   
3. First install dlib with using `conda-forge`

  ``` {powershell}
  conda install conda-forge::dlib
  ```

4. Install libraries listed on requirements.txt with using below command
  ``` {powershell}
  pip3 install -r ./requirements.txt
  ```

5. Install `pytorch` and `CUDA` if available.

    * If you have cuda, visit [Pytorch website](https://pytorch.org/get-started/locally/) to install cuda and pytorch.

       Below is example of using it.
      ``` {powershell}
      pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
      ```
    * If you don't have cuda, install `pytorch`
      ``` {powershell}
      pip3 install torch torchvision torchaudio
      ```

6. Run `mosaic_gui.py`
  ``` {powershell}
  python mosaic_gui.py
  ```
7. Enjoy!
   * Do not worry about CUDA because it will automatically set into CPU when there are no cuda available.
   * Use example images and folder on `mosaicv1` folder, which is `RedVelvet.jpg` and `FaceDB` respectively.

8. LIMITATIONS:
   
   `face_recognition` has poor performance in calculating embedding vectors.

    To solve these problems, fix the `tolerance` value on `mosaic_gui.py`. 
