# dejavu-DLapi

## How to deploy

This api has been dockerized. To deploy it, follow these three steps:
1. Install [Docker](https://docs.docker.com/engine/install/debian/) and [Docker Compose](https://docs.docker.com/compose/install/) according to Linux distribution.
2. Git clone this repo and enter the directory.
3. Run this command: `sudo bash run_docker.sh`, and wait for it to complete.
4. Nginx will listen to port 5000 by default, which can be changed in docker-compose.yml.
5. One can send GET request to it and get results.