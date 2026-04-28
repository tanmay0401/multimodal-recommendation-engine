FROM python:3.11-slim

# Create a non-root user 'user' with UID 1000
RUN useradd -m -u 1000 user

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Switch to the non-root user
USER user

# Set up environment variables
ENV HOME=/home/user
ENV PATH=/home/user/.local/bin:$PATH
ENV PYTHONPATH=$HOME/app
# Ensure Hugging Face cache uses the writable home directory
ENV HF_HOME=$HOME/.cache/huggingface

# Set working directory
WORKDIR $HOME/app

# Copy requirements and install them
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project as the non-root user
COPY --chown=user . $HOME/app

# Hugging Face Spaces expects port 7860
EXPOSE 7860

# Run the API on port 7860
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "7860"]
