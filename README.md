# dejavu-DLapi

## How to start

sudo systemctl start nginx

gunicorn --bind 0.0.0.0:5000 app:app --daemon