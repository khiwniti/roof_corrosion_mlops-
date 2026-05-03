# RunPod Serverless Setup Guide (Manual Web UI)

The RunPod API currently requires manual setup via the web UI. Follow these exact steps:

## Step 1: Create Serverless Template

Go to: https://www.runpod.io/console/serverless/templates

Click **"+ New Template"** and fill:

| Field | Value |
|-------|-------|
| **Template Name** | `roof-corrosion-serverless` |
| **Container Image** | `docker.io/khiwnitigetintheq/roof-corrosion-inference:latest` |
| **Container Disk (GB)** | `10` |
| **Volume Mount Path** | (leave empty) |

**Environment Variables** - Add each:
```
REGION=TH
PIPELINE=fm
PYTHONPATH=/app/src
PYTHONUNBUFFERED=1
MIN_CONFIDENCE=0.6
```

**Container Registry Credentials**:
- If your Docker Hub repo is public: leave empty
- If private: add your Docker Hub username/access token

Click **"Create Template"**

---

## Step 2: Create Serverless Endpoint

Go to: https://www.runpod.io/console/serverless

Click **"+ New Endpoint"**

### Basic Settings
| Field | Value |
|-------|-------|
| **Name** | `roof-corrosion-inference` |
| **Select Template** | Choose `roof-corrosion-serverless` |

### GPU Selection
Pick the **cheapest available** (our code doesn't use GPU, but RunPod requires one):
- Preferred: `NVIDIA RTX A2000` (~$0.12/hr)
- Alternative: `NVIDIA RTX A4000` (~$0.16/hr)
- Alternative: `Tesla T4` (~$0.15/hr)

### Worker Configuration
| Field | Value |
|-------|-------|
| **Workers Min** | `0` |
| **Workers Max** | `2` (your account quota allows 5 total, 3 used by EasyOCR) |
| **Idle Timeout** | `300` seconds |
| **FlashBoot** | ✅ Enable (faster cold starts) |

### Advanced Settings (click "Advanced")
| Field | Value |
|-------|-------|
| **Execution Timeout (ms)** | `60000` (1 minute) |
| **Scaler Type** | `QUEUE_DELAY` |
| **Scaler Value** | `4` |

Click **"Deploy"**

---

## Step 3: Get Endpoint ID

After deployment, you'll see the endpoint details page.

**Copy the Endpoint ID** (looks like `abc123def456`)

Add to your `.env` file:
```bash
echo "RUNPOD_ENDPOINT_ID=your_endpoint_id_here" >> /Users/admin/roof_corrosion_mlops/.env
```

---

## Step 4: Test the Endpoint

```bash
cd /Users/admin/roof_corrosion_mlops
source .env

curl -s -X POST "https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/runsync" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "lat": 13.7563,
      "lng": 100.5018,
      "address": "Silom Rd, Bangkok",
      "gsd": 0.3,
      "region": "TH"
    }
  }'
```

Expected response:
```json
{
  "status": "completed",
  "assessment": {
    "roof_area_m2": 150.0,
    "corrosion_percent": 15.0,
    "severity": "light"
  },
  "quote": {
    "currency": "THB",
    "total_amount": 72500.0
  }
}
```

---

## Troubleshooting

### "No workers available" error
- Your EasyOCR endpoint uses 3 workers (quota = 5)
- Set this endpoint to max 2 workers

### "Image pull failed" error
- Verify Docker Hub image exists: `docker pull docker.io/khiwnitigetintheq/roof-corrosion-inference:latest`
- Check if repo is public

### "NIM API key required" error
- Get free API key from https://build.nvidia.com
- Add to endpoint environment variables as `NVIDIA_API_KEY`

---

## Next Steps

1. After creating endpoint, add `RUNPOD_ENDPOINT_ID` to `.env`
2. Update GitHub Actions variable for CI/CD
3. Test the full flow from frontend → API → RunPod
