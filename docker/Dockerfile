FROM ulissigroup/ocpmodels:latest

# Reset the default user to `root` so that we can make changes
USER root

# Install new conda packages
RUN conda install seaborn
RUN conda clean -ity

# Set the default user back to `user` so that we can use Shifter
USER user
