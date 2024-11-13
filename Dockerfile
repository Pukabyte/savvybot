# Use Python 3.9 instead of 3.8
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download the spaCy language model
RUN python -m spacy download en_core_web_sm

# Copy the bot code and data files into the container
COPY . .

# Run the bot
CMD ["python", "bot.py"]
