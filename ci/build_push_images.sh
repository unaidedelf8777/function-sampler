#!/bin/bash

# Exit script on any error
set -e

cleanup() {
    echo "Removing Docker Buildx builder..."
    docker buildx rm cibw_builder
}

# Set trap to clean up resources on EXIT
trap cleanup EXIT

# for finding docker files.
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

# Check for repository argument
if [ -z "$1" ]; then
    echo "Error: No repository specified."
    echo "Usage: $0 <repository>"
    exit 1
fi

# Set the repository variable
REPO="$1"

# Check if the user is logged into Docker Hub
if ! docker info | grep -q "Username"; then
    echo "You are not logged into Docker Hub. Please log in to continue."
    docker login
    if [ $? -ne 0 ]; then
        echo "Docker login failed. Please check your credentials and try again."
        exit 1
    fi
fi

# Step 1: Install QEMU for cross-platform builds
echo "Installing QEMU packages for emulation..."
sudo apt-get update
sudo apt-get install -y qemu qemu-user-static qemu-user binfmt-support

# Step 2: Initialize Docker Buildx with proper binfmt_misc setup using the specific docker/binfmt image
echo "Setting up binfmt_misc for cross-platform builds..."
docker run --rm --privileged docker/binfmt:a7996909642ee92942dcd6cff44b9b95f08dad64


# Step 3: Create a new buildx builder instance
echo "Creating new Docker Buildx builder..."
docker buildx create --name cibw_builder --use

# Step 4: Start up the buildx builder
echo "Starting up the Docker Buildx builder..."
docker buildx inspect --bootstrap

# Define the build and push function for Docker images
build_and_push() {
    local dockerfile="${SCRIPT_DIR}/$1"
    local platform=$2
    local arch=$(echo $platform | cut -d '/' -f2)  # Extract only the architecture part
    local tag="${REPO}:${arch}"  # Use only the architecture in the tag

    echo "Building Docker image for platform ${platform}..."
    docker buildx build --platform ${platform} -f ${dockerfile} -t ${tag} --push .
}

# Step 5: Build and push Docker images for each architecture
echo "Building and pushing Docker images..."

# Build and push for x86_64
build_and_push "Dockerfile.cibw_x86_64" "linux/x86_64"

# Build and push for aarch64
build_and_push "Dockerfile.cibw_aarch64" "linux/aarch64"

echo "Docker images have been built and pushed successfully."

cleanup
