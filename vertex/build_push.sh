IMAGE_NAME=us-central1-docker.pkg.dev/bioptic-io/dl/asap-admet:ubuntu22.04-cu12.6.0-py311-pt2.4

docker build \
    -t $IMAGE_NAME \
    -f vertex/Dockerfile \
    .

docker push $IMAGE_NAME