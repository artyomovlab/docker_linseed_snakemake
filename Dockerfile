FROM rocker/r-ver:4.1

RUN apt-get update && \
  apt-get install -y libxml2-dev libcurl4-openssl-dev libssl-dev
RUN R -e "install.packages('remotes')"
RUN R -e "install.packages('collections')"
RUN R -e "remotes::install_github('ctlab/linseed')"
RUN R -e "install.packages('plotly')"
RUN R -e "install.packages('matlib')"
RUN R -e "install.packages('matrixcalc')"
RUN R -e "install.packages('optparse')"
RUN R -e "install.packages('yaml')"
RUN R -e "install.packages('nnls')"
RUN R -e "install.packages('Rcpp')"
RUN R -e "install.packages('RcppArmadillo')"
RUN R -e "install.packages('rmarkdown')"
RUN R -e "install.packages('uwot')"
RUN R -e "install.packages('ggpubr')"
RUN R -e "install.packages('dbscan')"
RUN R -e "install.packages('lsa')"
RUN R -e "install.packages('reshape2')"

RUN  apt-get update \
  && apt-get install -y wget \
  && rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# Install snakemake and other packages
COPY environment.yaml /app/environment.yaml
RUN /opt/conda/bin/conda update  --yes -n base -c defaults conda setuptools
RUN /opt/conda/bin/conda env update -n base --file /app/environment.yaml
RUN /opt/conda/bin/conda clean   --yes --all
RUN rm /app/environment.yaml

# Solve locale issues when running bash.
#   /bin/bash: warning: setlocale: LC_ALL: cannot change locale (en_US.UTF-8)
#
# It breaks conda version check in snakemake:
RUN apt-get clean && apt-get update && apt-get install -y locales && \
    echo "LC_ALL=en_US.UTF-8" >> /etc/environment  && \
    echo "en_US.UTF-8 UTF-8" >> /etc/locale.gen  && \
    echo "LANG=en_US.UTF-8" > /etc/locale.conf  && \
    locale-gen en_US.UTF-8

COPY app /app
RUN chmod +x /app/scripts/run_linseedv2.py