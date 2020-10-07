# Use NVIDIA CUDA as base image and run the same installation as in the other packages.
# The version of cudatoolkit must match those of the base image, see Dockerfile.pytorch
FROM ulissigroup/jovyan_base

# Configure container startup
ENTRYPOINT ["tini", "-g", "--"]

USER $NB_UID

########## Begin image-specific configurations ##########

# Install Jupyter Notebook, Lab, and Hub
# Do all this in a single RUN command to avoid duplicating all of the
# files across image layers when the permissions change
RUN conda install --quiet --yes \
    jupyter \
    jupyterhub \
    jupyterlab \
    jupyter_nbextensions_configurator && \
    jupyter nbextensions_configurator enable && \
    # Pip installations
    pip install jupyterthemes && \
    pip install git+https://github.com/rossant/ipycache.git && \
    conda clean --all -f -y && \
    # Generate a notebook server config
    jupyter notebook --generate-config && \
    # VIM bindings for Jupyter
    mkdir -p $(jupyter --data-dir)/nbextensions && \
    git clone https://github.com/lambdalisue/jupyter-vim-binding $(jupyter --data-dir)/nbextensions/vim_binding && \
    # Cleanup temporary files
    rm -rf $CONDA_DIR/share/jupyter/lab/staging && \
    rm -rf /home/$NB_USER/.cache/yarn && \
    # Correct permissions
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER

# Install dependencies for "baselines" repo
RUN conda config --prepend channels pytorch && \
    conda install --quiet --yes \
    cudatoolkit=10.1 \
    ase=3.19.* \
    pymatgen=2020.4.2 \
    pre-commit=2.2.* \
    pytorch=1.5.* \
    tensorboard=1.15.* \
    pyyaml=5.3.* \
    gpytorch \
    pytest && \
    # Clean up
    conda clean --all -f -y && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER

# Pip dependencies for "baselines" repo
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    demjson \
    Pillow \
    ray[tune] \
    torch-geometric==1.5.* \
    wandb \
    lmdb==0.98 \
    -f https://pytorch-geometric.com/whl/torch-1.5.0.html \
    torch-cluster==latest+cu101 \
    torch-scatter==latest+cu101 \
    torch-sparse==latest+cu101 \
    torch-spline-conv==latest+cu101 && \
    # Clean up
    conda clean --all -f -y && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER

# Install baselines
RUN pip install --no-cache-dir git+https://github.com/Open-Catalyst-Project/baselines.git@cfgp_gpu

# Install catalyst-acquisitions dependencies
RUN conda config --append channels lmmentel && \
    conda config --append channels plotly && \
    conda install --quiet --yes \
    mendeleev \
    tpot>=0.9.5 xgboost>=0.80 \
    seaborn \
    plotly>=4.1.1 chart-studio>=1.0.0 \
    shapely \
    fireworks \
    luigi>=2.8.9 \
    statsmodels>=0.9.0 \
    multiprocess>=0.70.5 \
    pymongo=3.8.0 \
    atomicwrites && \
    # Clean up
    conda clean --all -f -y && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER

# Install profilers to help with development
RUN conda install --quiet --yes pyinstrument line_profiler && \
    # Clean up
    conda clean --all -f -y && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER

# Install VIM
USER root
RUN apt-get update && apt-get install -yq --no-install-recommends vim && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
USER $NB_UID

########## End image-specific configurations ##########

EXPOSE 8888

# Switch back to jovyan to avoid accidental container runs as root
WORKDIR $HOME

USER root
RUN fix-permissions /home/$NB_USER

# Switch back to jovyan to avoid accidental container runs as root
USER $NB_UID

ENV NB_PREFIX /
CMD ["sh","-c", "jupyter notebook --notebook-dir=/home/jovyan --ip=0.0.0.0 --no-browser --allow-root --port=8888 --NotebookApp.token='' --NotebookApp.password='' --NotebookApp.allow_origin='*' --NotebookApp.base_url=${NB_PREFIX}"]
