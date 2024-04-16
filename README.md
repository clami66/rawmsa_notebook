## Rawmsa_disorder2 usage
```
Download docker image <DOCKERHUB-URL>

# Load image
sudo docker load -i <image-path>

# Run container
sudo docker run -v <path-to-input-fasta-folder>:/app/input_fasta -v <path-to-msa-folder>:/app/mounted_msas rawmsa_disorder2

# Retreive results
sudo docker cp <container-id>:/app/results .
```
