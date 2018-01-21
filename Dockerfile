# Base image
FROM python:3

# Author information
MAINTAINER artem.nikitin@skolkovotech.ru

# Set a working directory
WORKDIR Project

# Install latex.
RUN apt-get update && apt-get install -y texlive

# Install necessary libraries
RUN pip install numpy scipy matplotlib PIL

# Add necessary files. Good practice to do it at the end
# in order to avoid reinstallation of dependencies when files change
ADD Code ./Code
ADD Latex ./Latex
ADD Results ./Results
ADD run.sh ./

# Make run.sh executable
RUN chmod +x run.sh

VOLUME /Project/Results

CMD ./run.sh
