FROM ulissigroup/ocpmodels:latest

# Reset the default user to `root` so that we can make changes
USER root

# Install new conda packages
RUN conda config --append channels lmmentel
RUN conda config --append channels plotly
RUN conda install \
    seaborn \
    mendeleev \
    tpot>=0.9.5 xgboost>=0.80 \
    plotly>=4.1.1 chart-studio>=1.0.0 \
    shapely \
    fireworks \
    luigi>=2.8.9 \
    statsmodels>=0.9.0 \
    pytest=5.0.1 \
    multiprocess>=0.70.5
RUN conda install pymongo=3.8.0
RUN conda clean -ity

# Install GASpy
RUN git clone https://github.com/ulissigroup/GASpy.git $HOME/GASpy
RUN git clone https://github.com/ulissigroup/GASpy_regressions.git $HOME/GASpy/GASpy_regressions
ENV ENV PATH="$HOME/GASpy:$HOME/GASpy/GASpy_regressions:${PATH}"

# Set the default user back to `user` so that we can use Shifter
USER user
