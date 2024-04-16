# Use an existing Python image as a base
FROM python:3.8

# Set the working directory inside the container
WORKDIR /app
VOLUME /app/input_fasta
VOLUME /app/mounted_msas

# Copy the entire project directory into the container
COPY . .

RUN pip install -r requirements.txt

# Activate the conda environment
#SHELL ["conda", "run", "-n", "barebones", "/bin/bash", "-c"]

# Expose any ports the app is expecting
EXPOSE 8000

# Command to run the application
CMD ["python", "run_rawmsa_disorder.py"]