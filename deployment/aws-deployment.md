# AWS Deployment — InspectAI Inspection API

> **Verification status (honest):** The container image was **not** built or
> pushed, and **nothing was deployed to AWS**, because (a) no AWS credentials or
> AWS CLI are available in the build environment, and (b) the local Docker
> daemon was not running at build time. The FastAPI app itself was verified
> running locally via `uvicorn` (health + `/inspect` return correct results —
> see the project status report). The steps below are complete and ready to
> run, but are **UNVERIFIED** until executed with real credentials.

## What runs on AWS

The meaningful cloud component is the **InspectAI inspection API** (FastAPI,
`src/api/routes.py`), containerized via the repo `Dockerfile`. It:

- accepts an uploaded image (`POST /inspect`),
- runs the full OpenCV 5 pipeline + agentic decision loop server-side,
- returns structured results (status, score, regions, decision trace) and
  base64-encoded annotated/heatmap images,
- exposes `GET /health` for the platform health check.

The Streamlit UI can call this API by setting `INSPECTAI_API_URL`.

## Recommended path: AWS App Runner (simplest)

App Runner builds/runs a container with no cluster to manage, scales to a small
instance, and provides HTTPS + health checks out of the box.

### 1. Build and push the image to Amazon ECR

```bash
AWS_REGION=us-east-1
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REPO=inspectai

aws ecr create-repository --repository-name $REPO --region $AWS_REGION
aws ecr get-login-password --region $AWS_REGION \
  | docker login --username AWS --password-stdin \
    $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

docker build -t $REPO:latest .
docker tag $REPO:latest $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPO:latest
docker push $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPO:latest
```

### 2. Create the App Runner service

Use `deployment/apprunner.yaml` (reference config) or the console/CLI. The
service must:

- use the pushed ECR image,
- set container port **8080**,
- set health-check path **`/health`**,
- set env vars `INSPECTAI_MAX_UPLOAD_BYTES`, `INSPECTAI_MAX_AGENT_ITERS`.

```bash
aws apprunner create-service \
  --service-name inspectai \
  --source-configuration file://deployment/apprunner-source.json \
  --health-check-configuration 'Protocol=HTTP,Path=/health,Interval=10,Timeout=5,HealthyThreshold=1,UnhealthyThreshold=5' \
  --instance-configuration 'Cpu=1024,Memory=2048' \
  --region $AWS_REGION
```

After it reaches `RUNNING`, App Runner prints a public HTTPS URL. Verify:

```bash
curl https://<service-id>.$AWS_REGION.awsapprunner.com/health
```

## Alternative: Amazon ECS Fargate

Push the same image to ECR, then create an ECS Fargate service (0.5 vCPU /
1 GB is sufficient for this classical pipeline) behind an Application Load
Balancer with the target-group health check pointed at `/health`. This is more
moving parts than App Runner; prefer App Runner for the demo.

## Alternative: AWS Lambda + API Gateway

Feasible because the pipeline is CPU-light (~5–10 ms/image) and dependency
footprint is modest. Package with a container image for Lambda (the `opencv`
+ `numpy` wheels exceed the zip layer limit). Note the API Gateway payload
limit (10 MB) is compatible with the 8 MB upload cap. Cold starts add latency;
App Runner is smoother for an interactive demo.

## IAM permissions required

The **operator** (person deploying) needs:
`ecr:*` (or the create/push subset), `apprunner:CreateService`,
`apprunner:DescribeService`, `iam:PassRole` for the App Runner access role,
and `sts:GetCallerIdentity`.

The App Runner **instance role** needs **no** extra AWS permissions for this
app (it calls no other AWS services). An App Runner **ECR access role**
(`AWSAppRunnerServicePolicyForECRAccess`) is required to pull the image.

## Secrets & configuration

- **No credentials are stored in source.** The app reads only non-secret
  tuning via environment variables (`INSPECTAI_*`).
- If future features need AWS services, use the task/instance IAM role — never
  embedded keys.

## Estimated cost

- **App Runner**, 1 vCPU / 2 GB, scales to a minimum provisioned instance:
  roughly on the order of a few US dollars per day if left always-on; set a
  low `MinSize` and pause/delete the service when not demoing to minimize cost.
- **ECR** storage for one small image: negligible.
- **Do not leave the service running** after the demo to avoid ongoing charges.

> Per the project constraints, **no paid AWS infrastructure was created**. The
> operator must run the commands above explicitly.
