import multiprocessing

# Gunicorn configuration - Optimized for Render free tier (512MB)
bind = "0.0.0.0:5000"
workers = 1  # Single worker to save memory
worker_class = "sync"
worker_connections = 100  # Reduced from 1000
timeout = 120  # Increase from default 30 to 120 seconds
keepalive = 5
max_requests = 100  # Reduced from 1000 - restart worker more often
max_requests_jitter = 50
graceful_timeout = 30