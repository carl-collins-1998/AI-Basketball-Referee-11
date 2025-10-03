name: Build and Publish Docker image

on:
  push:
    branches:
      - main

jobs:
  build-and-publish:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
      id-token: write
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up QEMU (optional for multi-arch)
        uses: docker/setup-qemu-action@v2

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2

      - name: Login to GitHub Container Registry
        uses: docker/login-action@v2
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      # Optional: Docker Hub login if you prefer (requires DOCKERHUB_USERNAME & DOCKERHUB_TOKEN secrets)
      - name: Login to Docker Hub (optional)
        if: ${{ secrets.DOCKERHUB_USERNAME && secrets.DOCKERHUB_TOKEN }}
        uses: docker/login-action@v2
        with:
          registry: docker.io
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}

      - name: Build and push to GHCR
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: |
            ghcr.io/${{ github.repository_owner }}/ai-basketball-referee:${{ github.sha }}
            ghcr.io/${{ github.repository_owner }}/ai-basketball-referee:latest
          platforms: linux/amd64,linux/arm64

      - name: Optionally push tag to Docker Hub
        if: ${{ secrets.DOCKERHUB_USERNAME && secrets.DOCKERHUB_TOKEN }}
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: |
            ${{ secrets.DOCKERHUB_USERNAME }}/ai-basketball-referee:${{ github.sha }}
            ${{ secrets.DOCKERHUB_USERNAME }}/ai-basketball-referee:latest
          platforms: linux/amd64,linux/arm64