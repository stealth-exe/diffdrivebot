#!/bin/bash

# Configuration
IMAGE="diffdrivebot:harmonic-mac"
NAME="line_follower"

# Mount the repository this script lives in, so the container always sees the
# current checkout (works no matter where the repo was cloned).
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Check if the container exists (running or stopped)
if [ "$(docker ps -aq -f name="^/${NAME}$")" ]; then
    # Check if the container is currently running
    if [ "$(docker ps -q -f name="^/${NAME}$")" ]; then
        echo "Connecting to running container: ${NAME}..."
        docker exec -it "$NAME" bash -i
    else
        echo "Starting and connecting to stopped container: ${NAME}..."
        docker start -ai "$NAME"
    fi
else
    # Create and run a new container if it doesn't exist
    echo "Creating and starting new container: ${NAME}..."
    docker run -it \
        --name "$NAME" \
        --shm-size=1g \
        -p 6767:6080 \
        --env ROS_DOMAIN_ID=0 \
        --env GZ_VERSION=harmonic \
        -v "${REPO}:/tmp/diffdrivebot" \
        "$IMAGE" bash -i
fi