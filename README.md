## rawmsa_disorder: Evolutionary sequence information to identify protein disorder
```
                                              _ _                   _           
                                             | (_)                 | |          
  _ __ __ ___      ___ __ ___  ___  __ _   __| |_ ___  ___  _ __ __| | ___ _ __ 
 | '__/ _` \ \ /\ / / '_ ` _ \/ __|/ _` | / _` | / __|/ _ \| '__/ _` |/ _ \ '__|
 | | | (_| |\ V  V /| | | | | \__ \ (_| || (_| | \__ \ (_) | | | (_| |  __/ |   
 |_|  \__,_| \_/\_/ |_| |_| |_|___/\__,_| \__,_|_|___/\___/|_|  \__,_|\___|_|   
                                      ______                                    
                                     |______|                                   
```
## Installation

1. Create new conda environment: `conda create -n rawmsa_disorder python==3.9`
2. Activate the conda environment: `conda activate ramwsa_disorder`
3. Install necessary packages with pip: `python -m pip install -f docker/requirements.txt`
4. Set up the following environmental variable: `export RAWMSA_PATH=$(pwd)` from the `rawmsa_disorder/` directory

## Run

1. Launch the mercury server: `mercury run notebooks/rawmsa_disorder.ipynb`
2. Open the browser at: [http://127.0.0.1:8000/app/rawmsa_disorder](http://127.0.0.1:8000/app/rawmsa_disorder)
