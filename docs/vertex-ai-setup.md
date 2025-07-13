# Vertex AI Setup Guide

This guide will help you configure Google Cloud Vertex AI to work with Zen MCP Server. Vertex AI provides enterprise-grade access to Google's Gemini models with additional security, compliance, and regional deployment options.

## Prerequisites

- A Google Cloud Platform (GCP) account
- A GCP project with billing enabled
- `gcloud` CLI installed on your machine ([Installation Guide](https://cloud.google.com/sdk/docs/install))

## Quick Start

### 1. Enable Vertex AI API

```bash
# Set your project ID
export PROJECT_ID="your-project-id"

# Set as default project
gcloud config set project $PROJECT_ID

# Enable Vertex AI API
gcloud services enable aiplatform.googleapis.com

# Enable ML API (required for model discovery)
# This will prompt you to enable the API if not already enabled
gcloud ai-platform models list
# When prompted: "API [ml.googleapis.com] not enabled on project [<project>]. Would you like to enable and retry (this will take a few minutes)? (y/N)?"
# Answer: y
```

### 2. Set Up Authentication

You have three authentication options:

#### Option A: Application Default Credentials (Recommended for Development)

```bash
# Login with your Google account
gcloud auth application-default login

# This creates credentials at:
# - Linux/Mac: ~/.config/gcloud/application_default_credentials.json
# - Windows: %APPDATA%\gcloud\application_default_credentials.json
```

#### Option B: Service Account (Recommended for Production)

```bash
# Create a service account
gcloud iam service-accounts create zen-vertex-ai \
    --display-name="Zen MCP Vertex AI Service Account"

# Grant necessary permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:zen-vertex-ai@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"

# Create and download key
gcloud iam service-accounts keys create ~/vertex-ai-key.json \
    --iam-account=zen-vertex-ai@$PROJECT_ID.iam.gserviceaccount.com

# Set environment variable
export GOOGLE_APPLICATION_CREDENTIALS=~/vertex-ai-key.json
```

#### Option C: Workload Identity (For GKE/Cloud Run)

If running on Google Cloud infrastructure, you can use Workload Identity to avoid managing service account keys.

### 3. Configure Zen MCP Server

Add these variables to your `.env` file:

```env
# Required: Your GCP project ID
VERTEX_AI_PROJECT_ID=your-project-id

# Optional: Specify region for Gemini models (defaults to us-central1)
# See available regions: https://cloud.google.com/vertex-ai/docs/general/locations
VERTEX_AI_LOCATION=us-central1

# Optional: Specify region for Claude models (defaults to us-east5)
# Claude models are typically only available in us-east5
VERTEX_AI_CLAUDE_LOCATION=us-east5
```

### 4. Verify Setup

Test your configuration:

```bash
# Check authentication
gcloud auth application-default print-access-token

# Test Vertex AI access with current model names
curl -X POST \
  -H "Authorization: Bearer $(gcloud auth application-default print-access-token)" \
  -H "Content-Type: application/json" \
  "https://${VERTEX_AI_LOCATION:-us-central1}-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/${VERTEX_AI_LOCATION:-us-central1}/publishers/google/models/gemini-2.5-flash:generateContent" \
  -d '{
    "contents": [{
      "role": "user",
      "parts": [{"text": "Hello, respond with just SUCCESS"}]
    }]
  }'

# Expected response should include: "text": "SUCCESS"
```

## Available Models

Zen MCP Server supports these Vertex AI models:

### Gemini Models (Generally Available)

| Model ID | Alias | Context Window | Description |
|----------|-------|----------------|-------------|
| `gemini-2.5-pro` | `vertex-gemini-pro`, `vertex-pro` | 1M tokens | Latest Gemini Pro with thinking mode support |
| `gemini-2.5-flash` | `vertex-gemini-flash`, `vertex-flash` | 1M tokens | Latest Gemini Flash with thinking mode support |
| `gemini-2.0-flash` | `vertex-gemini-2-flash`, `vertex-2-flash` | 1M tokens | Newest multimodal model with thinking mode |
| `gemini-2.0-flash-lite` | `vertex-gemini-2-flash-lite`, `vertex-2-flash-lite` | 1M tokens | Cost-optimized model with low latency |
| `gemini-2.5-flash-lite` | `vertex-gemini-flash-lite`, `vertex-flash-lite` | 1M tokens | Most cost-effective for high throughput |
| `gemini-1.5-pro-002` | `vertex-gemini-1.5-pro`, `vertex-1.5-pro` | 2M tokens | Legacy Gemini 1.5 Pro (no thinking mode) |
| `gemini-1.5-flash-002` | `vertex-gemini-1.5-flash`, `vertex-1.5-flash` | 1M tokens | Legacy Gemini 1.5 Flash (no thinking mode) |

### Claude Models (If Available in Your Project)

Some GCP projects may have access to Anthropic Claude models through Vertex AI:

| Model ID | Alias | Context Window | Description |
|----------|-------|----------------|-------------|
| `claude-sonnet-4@20250514` | `vertex-claude-sonnet-4`, `vertex-sonnet-4` | 200K tokens | Balanced performance and capability |
| `claude-opus-4@20250514` | `vertex-claude-opus-4`, `vertex-opus-4` | 200K tokens | Maximum capability and performance |
| `claude-3.7-sonnet` | `vertex-claude-37-sonnet`, `vertex-claude-3.7-sonnet` | 200K tokens | Extended thinking for agentic coding |
| `claude-3.5-sonnet-v2` | `vertex-claude-35-sonnet-v2`, `vertex-claude-3.5-sonnet-v2` | 200K tokens | State-of-the-art software engineering |
| `claude-3.5-sonnet` | `vertex-claude-35-sonnet`, `vertex-claude-3.5-sonnet` | 200K tokens | Advanced coding and analysis |
| `claude-3.5-haiku` | `vertex-claude-35-haiku`, `vertex-claude-3.5-haiku` | 200K tokens | Fast and cost-effective |
| `claude-3-haiku` | `vertex-claude-3-haiku`, `vertex-haiku-3` | 200K tokens | Fastest vision and text model |

**Note**: Claude model availability depends on your GCP project's access permissions and region. Claude models are typically only available in the `us-east5` region, which is why Zen MCP Server uses a separate `VERTEX_AI_CLAUDE_LOCATION` configuration.

## Regional Availability

Vertex AI models are available in multiple regions. Choose the region closest to your users for best performance:

### Recommended Regions
- **Americas**: `us-central1`, `us-east1`, `us-west1`
- **Europe**: `europe-west1`, `europe-west4`
- **Asia Pacific**: `asia-southeast1`, `asia-northeast1`

Check [current regional availability](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations) for the most up-to-date list.

## Model Restrictions

You can restrict which Vertex AI models can be used:

```env
# Only allow specific models
VERTEX_AI_ALLOWED_MODELS=vertex-pro,gemini-1.5-pro-002

# Only allow flash model for cost control
VERTEX_AI_ALLOWED_MODELS=vertex-flash
```

## Cost Management

### Pricing Overview
- Vertex AI charges per 1,000 characters (approximately 750 tokens)
- Pricing varies by model and region
- See [Vertex AI Pricing](https://cloud.google.com/vertex-ai/pricing#generative_ai_models) for current rates

### Cost Control Tips

1. **Use Flash models for routine tasks**
   ```env
   DEFAULT_MODEL=vertex-flash
   ```

2. **Set up budget alerts**
   ```bash
   gcloud billing budgets create \
     --billing-account=BILLING_ACCOUNT_ID \
     --display-name="Vertex AI Budget" \
     --budget-amount=100 \
     --threshold-rule=percent=50 \
     --threshold-rule=percent=90
   ```

3. **Monitor usage**
   ```bash
   # View recent API calls
   gcloud logging read "resource.type=aiplatform.googleapis.com/Model" \
     --limit=10 \
     --format=json
   ```

## Security Best Practices

### 1. Principle of Least Privilege

Only grant necessary permissions:

```bash
# Minimal role for using models
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:zen-vertex-ai@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"

# Additional role if creating endpoints
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:zen-vertex-ai@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/aiplatform.admin"
```

### 2. VPC Service Controls (Optional)

For enhanced security, configure VPC-SC:

```bash
# Create VPC-SC perimeter (enterprise only)
gcloud access-context-manager perimeters create vertex-ai-perimeter \
    --title="Vertex AI Perimeter" \
    --resources=projects/$PROJECT_ID \
    --restricted-services=aiplatform.googleapis.com
```

### 3. Private Endpoints (Optional)

Use Private Service Connect for internal access:

```bash
# Reserve IP range for private endpoint
gcloud compute addresses create vertex-ai-range \
    --global \
    --purpose=VPC_PEERING \
    --prefix-length=16 \
    --network=default
```

## Troubleshooting

### Authentication Issues

```bash
# Check current credentials
gcloud auth list

# Verify application default credentials
gcloud auth application-default print-access-token

# Reset credentials if needed
gcloud auth application-default revoke
gcloud auth application-default login
```

### API Not Enabled

```bash
# Check if API is enabled
gcloud services list --enabled | grep aiplatform

# Enable if missing
gcloud services enable aiplatform.googleapis.com
```

### Permission Errors

```bash
# Check IAM permissions
gcloud projects get-iam-policy $PROJECT_ID \
    --flatten="bindings[].members" \
    --filter="bindings.members:serviceAccount:*"

# Add missing permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="user:your-email@example.com" \
    --role="roles/aiplatform.user"
```

### Region Not Available

If you get region errors, check available locations:

```bash
# List available locations
gcloud ai-platform locations list
```

## Advanced Configuration

### Using Model Garden Models

Vertex AI Model Garden provides access to many open-source and third-party models. To use these:

1. Browse available models in the [Model Garden](https://console.cloud.google.com/vertex-ai/model-garden)
2. Deploy the model to an endpoint
3. Configure Zen MCP Server to use the endpoint (requires custom provider implementation)

### Batch Prediction Support

For processing large volumes of data, consider batch predictions:

```python
# Example batch prediction setup (not yet supported in Zen MCP)
batch_prediction_job = aiplatform.BatchPredictionJob.create(
    job_display_name="zen-batch-job",
    model_name=model.resource_name,
    instances_format="jsonl",
    gcs_source="gs://your-bucket/input.jsonl",
    gcs_destination_prefix="gs://your-bucket/output",
)
```

## Integration Examples

### Using Vertex AI with Zen MCP

```bash
# Ask Claude to use Vertex AI models
"Use vertex-pro to analyze this architecture design"
"Think deeply about this problem using vertex-gemini-1.5-pro"
"Use vertex-flash for a quick code review"
"Use vertex-claude-sonnet-4 for advanced reasoning"
```

### Auto Mode with Vertex AI

When `DEFAULT_MODEL=auto`, Claude will consider Vertex AI models alongside others:

```env
# Enable auto mode
DEFAULT_MODEL=auto

# Configure multiple providers
VERTEX_AI_PROJECT_ID=your-project-id
GEMINI_API_KEY=your-gemini-key
OPENAI_API_KEY=your-openai-key
```

## Monitoring and Logging

### View API Usage

```bash
# Recent API calls
gcloud logging read \
    'resource.type="aiplatform.googleapis.com/Model" AND 
     resource.labels.model_id=~"gemini"' \
    --limit=20 \
    --format="table(timestamp,jsonPayload.model_name,jsonPayload.prompt_token_count,jsonPayload.completion_token_count)"
```

### Set Up Alerts

```bash
# Create log-based metric
gcloud logging metrics create vertex-ai-errors \
    --description="Vertex AI API errors" \
    --log-filter='resource.type="aiplatform.googleapis.com/Model" AND severity="ERROR"'

# Create alert policy
gcloud alpha monitoring policies create \
    --notification-channels=CHANNEL_ID \
    --display-name="Vertex AI Error Rate" \
    --condition-display-name="Error rate too high" \
    --condition-filter='metric.type="logging.googleapis.com/user/vertex-ai-errors"'
```

## Support Resources

- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- [Gemini API Migration Guide](https://cloud.google.com/vertex-ai/generative-ai/docs/migrate-from-palm)
- [Vertex AI Support](https://cloud.google.com/vertex-ai/docs/support/getting-support)
- [GCP Status Page](https://status.cloud.google.com/)

## Next Steps

1. **Test your setup**: Run Claude with Zen MCP and try: `"Use vertex-pro to explain how this code works"`
2. **Monitor costs**: Set up budget alerts and review usage regularly
3. **Optimize performance**: Choose appropriate regions and models for your use case
4. **Explore Model Garden**: Access additional models beyond Gemini

For issues or questions, please open an issue on the [Zen MCP Server GitHub repository](https://github.com/BeehiveInnovations/zen-mcp-server/issues).