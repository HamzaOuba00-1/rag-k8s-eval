# Deploying DocTruth on Cloud Run

One Cloud Run service in `us-central1` serves the React front-end and the API from a
single image (`Dockerfile`). Everything below stays inside Google Cloud's free tier
for a low-traffic portfolio app; the budget alert is there to catch a mistake, it
does not cap spending.

| Resource | Setting | Why |
|---|---|---|
| Cloud Run | 1 vCPU, 1 GiB, min 0 / max 1 instance, concurrency 8 | The app process uses about 600 MB once the indexes and the ONNX model are loaded; one instance keeps the per-visitor quota in one place |
| Artifact Registry | one Docker repository, cleanup policy keeps the latest image | Image storage is free up to 0.5 GB |
| Cloud Build | default pool (e2-standard-2) | Free build-minutes cover a build of a few minutes |
| Cloud Storage | one bucket, one small object per day (quota counter), deleted after 30 days | |
| Secret Manager | one secret, the Groq API key | Never in the image, the repository or an environment file |

Commands are for Windows PowerShell 5.1 with the Google Cloud CLI. gcloud writes its
progress to stderr: PowerShell may show it in red even when the command succeeds.
Check the last lines and the exit code (`$LASTEXITCODE`, 0 = success).

## 0. Variables

```powershell
$PROJECT = "<your-project-id>"
$REGION  = "us-central1"
$REPO    = "doctruth"
$SERVICE = "doctruth"
$BUCKET  = "${PROJECT}-doctruth-quota"
$SA      = "doctruth-run@${PROJECT}.iam.gserviceaccount.com"
$IMAGE   = "${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/app"
gcloud config set project $PROJECT
```

To avoid retyping them, put these lines (plus `$BILLING`, see section 1) in
`deploy/env.ps1`, which Git ignores, and load them in each new session with
`. .\deploy\env.ps1`.

## 1. One-time setup

```powershell
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com `
  secretmanager.googleapis.com storage.googleapis.com billingbudgets.googleapis.com

# Image repository, keeping only the latest image
gcloud artifacts repositories create $REPO --repository-format=docker --location=$REGION
gcloud artifacts repositories set-cleanup-policies $REPO --location=$REGION `
  --policy=deploy/ar-cleanup-policy.json --no-dry-run

# Service identity of the app, with access to its own bucket and secret only
gcloud iam service-accounts create doctruth-run --display-name="DocTruth Cloud Run"

# Quota counter bucket: private, objects deleted after 30 days
gcloud storage buckets create "gs://${BUCKET}" --location=$REGION --uniform-bucket-level-access `
  --public-access-prevention --lifecycle-file=deploy/quota-bucket-lifecycle.json
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" `
  --member="serviceAccount:${SA}" --role=roles/storage.objectUser
```

Budget alert (e-mails at 50%, 90% and 100% of 1 USD; it does not stop anything):

```powershell
gcloud billing accounts list
$BILLING = "<billing-account-id>"
gcloud billing budgets create --billing-account=$BILLING --display-name="doctruth-1usd" `
  --budget-amount=1USD --filter-projects="projects/${PROJECT}" `
  --threshold-rule=percent=0.5 --threshold-rule=percent=0.9 --threshold-rule=percent=1.0
```

## 2. Build the image

The build context is the repository root, filtered by `.gcloudignore` (same rules as
`.dockerignore`): only the files the image needs are uploaded. They must exist
locally first:

```powershell
python -m ragk8s app-bundle          # app_data/bundle.json from results/
Test-Path data/index/fixed/dense.faiss, models/onnx/embedder/model.onnx, app_data/bundle.json
$TAG = git rev-parse --short HEAD
gcloud builds submit --tag "${IMAGE}:${TAG}"
```

## 3. Deploy

First deployment without the Groq key: demo mode (live retrieval, answers from the
evaluation runs). No Groq call can be made.

```powershell
gcloud run deploy $SERVICE --image "${IMAGE}:${TAG}" --region=$REGION `
  --service-account=$SA --allow-unauthenticated `
  --cpu=1 --memory=1Gi --min-instances=0 --max-instances=1 --concurrency=8 --timeout=60 `
  --set-env-vars="DOCTRUTH_QUOTA_BUCKET=${BUCKET}"
```

The command prints the service URL. Check it:

```powershell
$URL = gcloud run services describe $SERVICE --region=$REGION --format="value(status.url)"
Measure-Command { Invoke-RestMethod "${URL}/api/health" } | Select-Object TotalSeconds   # first call: cold start
Invoke-RestMethod "${URL}/api/health"
```

## 4. Turn on live answers (after the evaluation is complete)

The app and the evaluation share the Groq account's daily token limit, so live
answers are enabled only once every evaluation run is done. Use a dedicated key
(Groq console, named for example `doctruth-app`), so it can be revoked on its own.

```powershell
gcloud secrets create groq-api-key --replication-policy=automatic
$secure = Read-Host "Groq API key" -AsSecureString
$plain = [Runtime.InteropServices.Marshal]::PtrToStringBSTR([Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure))
$plain | gcloud secrets versions add groq-api-key --data-file=-
Remove-Variable plain, secure
gcloud secrets add-iam-policy-binding groq-api-key `
  --member="serviceAccount:${SA}" --role=roles/secretmanager.secretAccessor
gcloud run services update $SERVICE --region=$REGION --set-secrets="GROQ_API_KEY=groq-api-key:latest"
```

Quota settings can be changed without a rebuild, for example
`gcloud run services update $SERVICE --region=$REGION --update-env-vars=DOCTRUTH_DAILY_LIMIT=50`.

## 5. Update

```powershell
python -m ragk8s app-bundle
$TAG = git rev-parse --short HEAD
gcloud builds submit --tag "${IMAGE}:${TAG}"
gcloud run deploy $SERVICE --image "${IMAGE}:${TAG}" --region=$REGION
```

## 6. Take it down

```powershell
gcloud run services delete $SERVICE --region=$REGION
gcloud artifacts repositories delete $REPO --location=$REGION
gcloud storage rm --recursive "gs://${BUCKET}"
gcloud secrets delete groq-api-key
```
