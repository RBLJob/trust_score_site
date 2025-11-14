import multiprocessing

# Gunicorn configuration
bind = "0.0.0.0:5000"
workers = 1
worker_class = "sync"
worker_connections = 1000
timeout = 120  # Increase from default 30 to 120 seconds
keepalive = 5
max_requests = 1000
max_requests_jitter = 100
graceful_timeout = 30
