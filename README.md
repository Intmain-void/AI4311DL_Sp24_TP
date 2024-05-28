# DL_Sp24_TermProject_final       

GIST spring semester Deep learning term project

---

## How to run `mosaicv1`

1. Create an environment with conda with python version 3.8
  ``` {powershell}
  conda create -n DLTerm python=3.8
  ```
2. First install dlib with using `conda-forge`

  ``` {powershell}
  conda install conda-forge::dlib
  ```

3. Activate created environment and install libraries listed on requirements.txt with using below command

  ``` {powershell}
  pip3 install -r ./requirements.txt
  ```



파일 실행을 하기 위해서는 Requirements에 있는 library 설치가 필수입니다.

mosaic_gui.py를 실행하면, 발표에서 보여준 gui를 실행시킬 수 있습니다.
예시 파일로 RedVelvet.jpg와 faceDB 폴더가 준비되어 있습니다.
