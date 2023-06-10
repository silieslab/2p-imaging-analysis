# 2p-imaging-analysis
Basic analysis for calcium imaging in Python in Silies lab.


COMPUTER INSTALLATIONS:
----------------------
ENVIRONMENT MANAGER
- Download and install anaconda (or miniconda) from: https://docs.anaconda.com/anaconda/install/

GIT
- Download and install git from: https://git-scm.com/

FOR MOTION CORRECTION:
---------------------
There are several options so far untill we decide for a general one

IMAGEJ (Option for image alignment)
- Download and install git from https://imagej.net/software/fiji/
- Install following plugins: Image_Stabilizer.class and Image_Stabilizer_Log_Applier.class
  from here https://www.cs.cmu.edu/~kangli/code/Image_Stabilizer.html

SIMA (Option for image alignment)
- A package written in python 2.7. Tricky to install and deprecated: http://www.losonczylab.org/sima/1.3/install.html


ENVIRONMENT REQUISITES:
----------------------

INITIALIZE ENVIRONMENT
The anaconda prompt or the git bash terminals can be used.
In the git bash though, anaconda environments need to be set as source, running:
source C:/Users/[USERNAME]/anaconda3/Scripts/activate

In any of those terminals, follow the commands:

Setting the environment from the yml file:
- conda env create -f environment.yml
This will create an environment from parameters stored in the environment.yml file.

Alternatively, you can create a new environment and install each required package individually:
- conda create --name 2p_analysis python=3.7
- activate 2p_analysis (before installing any package)
  (INSTALL PACKAGES)
- pip install roipoly
- pip install numpy
- pip install pandas
- pip install matplotlib
- (other packages might be needed)

OPTIONAL PACKAGES
- pip install jupyterlab ( or just pip install notebook)

OPEN A JUPYTER NOTEBOOK in JUPYTER LAB
- Typer "jupyter-lab" (or jupyter notebook) in the activated environment prompt
- Look for the notebook file (.ipynb) and open it

OPTIONALLY, OTHER TEXT EDITORS CAN BE USED THAT HANDLE COMPLEX FILE TYPES:
- Atom: https://atom.io/
- VSCode: https://code.visualstudio.com/ (Recomended)

OR SIMPLER TEXT EDITORS, SUCH US:
- Vim (no installation needed)
- Nano (no installation needed)

RECOMENDED FOLDER STRUCTURE FOR ANALYSIS:
----------------------------------------

In order to store your raw data and subsequent files created during the analysis, the following folder structure is recommended.
Ennumeration and identations represent folders, subfolders and files inside.

    1. Chosen name for the experimental folder (e.g "Tm9_recordings" or "Tm9_luminances")
        1.1. "raw-data" (fixed folder name)
            1.1.1 "aligned-data" (fixed folder name)
                1.1.1.1 Folder with date+userID+flyID (e.g. "2021_09_30_seb_fly1", from the microscope computer)
                    1.1.1.1.1 TSeries (subfolder with recorded TIFFs)
                        1.1.1.1.1.1 Motion aligned TIFF stack (generated after motion correction)
                        1.1.1.1.1.2 Corresponding stim_output file to this TSeries (e.g."_stimulus_output_DATE_TIME")

        1.2 "analyzed-data" (fixed folder name)
            1.2.1 Stimtype (e.g. "LocalCircle_5secON_5sec_OFF_120deg_10sec", folder created by the code)
                1.2.1.1 Defined genotype (e.g."ExpLine", folder created by the code)
                    1.2.1.1.1 Pickle files (files generated and saved during the analysis)
        1.3 "stimulus-types" (fixed folder name)
            1.3.1 Different stimulus input files used during the experiment (STIMULUS_NAME.txt files)

    2. "data-save-vars.txt"* (fixed name, text file that lists the variables that want to save in a pickle file)
    3. "roi-save-vars.txt"* (fixed name, text file that lists the variables that want to save in a pickle file)

* Example txt files are found in the txt files folder of the repo
