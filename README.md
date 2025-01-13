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
## Usage (Docker)
```
Download docker image <DOCKERHUB-URL>

# Load image
sudo docker load -i <image-path>

# Run container
sudo docker run -v <path-to-input-fasta-folder>:/app/input_fasta -v <path-to-msa-folder>:/app/mounted_msas rawmsa_disorder2

# Retreive results
sudo docker cp <container-id>:/app/results .
```
## Usage (python)
```
# Clone repo
git clone https://github.com/clami66/rawmsa_disorder

# Run rawmsa_disorder
python docker/run_rawmsa_disorder.py --config config.txt
```
Results will be saved in .results/