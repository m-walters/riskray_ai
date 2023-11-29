# syntax=docker/dockerfile:1
# <><><><><><><><><><><><><><><><><><><><><>
# RiskRay tune image
# <><><><><><><><><><><><><><><><><><><><><>
# Pull base 3.9 image
FROM python:3.9-slim-bullseye

# Update packages
RUN apt-get update

# Set up a venv
# This is overkill, but still fun
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install the tune requirements
RUN python3 -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Navigate to build directory
WORKDIR /home/riskray

# Copy contents
COPY ai ai

# Tune session's config file -- Make this an argument
ARG TUNE_CONFIG="${WORKDIR}/ai/config/test0.yml"

# Add riskray source to python path
ENV PYTHONPATH="${PYTHONPATH}:${WORKDIR}"

# Set some python environment variables
# First, don't write .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
# Don't have python buffer outputs to stdout and stderr (I think)
ENV PYTHONUNBUFFERED=1

# RiskRay variables
ENV STORE_PATH=/home/riskray/ai/store
# Toggle our Keras NN model to be Sequential or Mixed Data
ENV KERAS_SEQUENTIAL=false
ENV USE_GPU=false
# More efficient memory usage for retrieving data from our dataset
ENV USE_GENERATORS=true

# RayTune
ENV TUNE_RESULT_DIR="$STORE_PATH/experiments"
ENV TUNE_RESULT_DELIM='/'

# Matplotlib -- No GUI
MPLBACKEND="agg"

# Make port 80 available for connecting to this container
EXPOSE 80

# Random seed
ENV SEED = 8675309

CMD ["python", "-m", "ai/training/tune.py", "--non-interactive", "$TUNE_CONFIG"]
