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

4. Activate created environment and install libraries listed on requirements.txt with using below command
  ``` {powershell}
  pip3 install -r ./requirements.txt
  ```

5. Run `mosaic_gui.py`
  ``` {powershell}
  python mosaic_gui.py
  ```
6. Enjoy!
   * Do not worry about CUDA because it will automatically set into CPU when there are no cuda available.
   * Use example images and folder on `mosaicv1` folder, which is `RedVelvet.jpg` and `FacdDB` respectively.

7. LIMITATIONS:
   `face_recognition` has poor performance in calculating embedding vectors.

    To solve these problems, fix the `tolerance` value on `mosaic_gui.py`. 
