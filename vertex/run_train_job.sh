chmod +x ./vertex/build_push.sh
./vertex/build_push.sh

gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=vladvin-asap-admet-challenge \
  --config=vertex/train-job-spec.yaml