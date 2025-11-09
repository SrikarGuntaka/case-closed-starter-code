# Case Closed Agent Dockerfile
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy dependency list first (for caching)
COPY requirements.txt .

# Install dependencies (CPU only, no cache)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Set default port for judge engine
ENV PORT=5008

# Expose the correct port
EXPOSE 5008

# Run your Flask agent
CMD ["python", "agent.py"]
