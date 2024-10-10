# Use the official miniconda3 image as the base image
FROM continuumio/miniconda3

# Set the working directory
WORKDIR /app

# Create the environment using conda
RUN conda create -n transhla python=3.9 -y

# Activate the environment and install packages via conda
RUN /bin/bash -c "source activate transhla && \
    conda install -n transhla -c conda-forge -c defaults numpy \
        pandas \
        matplotlib \
        tqdm \
        fair-esm \
        scikit-learn=1.3.0 \
        transformers \
        seaborn \
        pip -y"

# Install pip packages
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

# Set the default command to activate the environment and open a bash shell
CMD [ "/bin/bash", "-c", "source activate transhla && bash"]
