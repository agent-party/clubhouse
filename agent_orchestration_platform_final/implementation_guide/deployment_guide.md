# Deployment Guide

## Overview

This guide provides instructions for deploying the Agent Orchestration Platform in various environments. Following the quality-first principles, the deployment process emphasizes reliability, security, and observability.

## Deployment Architectures

### Development Environment

Lightweight setup for local development:

```
┌──────────────────┐     ┌──────────────┐     ┌───────────────┐
│                  │     │              │     │               │
│  MCP Server      │────▶│  Core        │────▶│  Neo4j        │
│                  │     │  Services    │     │  (Single Node)│
└──────────────────┘     └──────┬───────┘     └───────────────┘
                                │
                                ▼
                         ┌──────────────┐
                         │              │
                         │  Kafka       │
                         │  (Single Node)│
                         └──────────────┘
```

### Production Environment

High-availability setup for production:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│             │     │             │     │             │
│  MCP Server │     │  MCP Server │     │  MCP Server │
│  Instance 1 │     │  Instance 2 │     │  Instance 3 │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                   │
       └───────────────────┼───────────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │             │
                    │  Load       │
                    │  Balancer   │
                    └──────┬──────┘
                           │
                           ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│             │     │             │     │             │
│  Core       │     │  Core       │     │  Core       │
│  Services 1 │     │  Services 2 │     │  Services 3 │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                   │
       └───────────────────┼───────────────────┘
                           │
                           ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│             │     │             │     │             │
│ Kafka Broker│     │ Kafka Broker│     │ Kafka Broker│
│     1       │     │     2       │     │     3       │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                   │
       └───────────────────┼───────────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │             │
                    │  Neo4j      │
                    │  Cluster    │
                    └─────────────┘
```

## Infrastructure Requirements

### Development Environment

| Component | Resource Requirements | Notes |
|-----------|----------------------|-------|
| MCP Server | 2 vCPUs, 4GB RAM | Single instance |
| Core Services | 2 vCPUs, 4GB RAM | Co-located with MCP Server |
| Neo4j | 2 vCPUs, 4GB RAM | Single node |
| Kafka | 2 vCPUs, 2GB RAM | Single broker |
| **Total** | **8 vCPUs, 14GB RAM** | |

### Production Environment

| Component | Resource Requirements | Notes |
|-----------|----------------------|-------|
| MCP Server | 4 vCPUs, 8GB RAM per instance | 3+ instances |
| Core Services | 4 vCPUs, 8GB RAM per instance | 3+ instances |
| Neo4j | 8 vCPUs, 32GB RAM | 3+ node cluster |
| Kafka | 4 vCPUs, 8GB RAM per broker | 3+ brokers |
| Load Balancer | 2 vCPUs, 4GB RAM | |
| **Total** | **58+ vCPUs, 142+ GB RAM** | Minimum production setup |

## Local Development Setup

### Prerequisites

- Docker and Docker Compose
- Python 3.9+
- OpenAI API key

### Quick Start

1. Clone the repository:

```bash
git clone https://github.com/your-org/agent-orchestration-platform.git
cd agent-orchestration-platform
```

2. Create a `.env` file:

```
OPENAI_API_KEY=your_api_key_here
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
```

3. Start the development environment:

```bash
docker-compose -f docker-compose.dev.yml up
```

4. Install the Python dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

5. Run the platform:

```bash
python -m agent_orchestration.main
```

6. Verify the installation:

```bash
curl http://localhost:8000/health
```

## Docker Deployment

### Docker Compose Configuration

```yaml
# docker-compose.yml
version: '3.8'

services:
  mcp-server:
    image: agent-orchestration/mcp-server:latest
    build:
      context: .
      dockerfile: dockerfiles/mcp-server.Dockerfile
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - NEO4J_URI=neo4j://neo4j:7687
      - NEO4J_USERNAME=${NEO4J_USERNAME}
      - NEO4J_PASSWORD=${NEO4J_PASSWORD}
      - KAFKA_BOOTSTRAP_SERVERS=kafka:9092
    depends_on:
      - neo4j
      - kafka
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 5

  neo4j:
    image: neo4j:4.4
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      - NEO4J_AUTH=${NEO4J_USERNAME}/${NEO4J_PASSWORD}
      - NEO4J_dbms_memory_heap_max__size=2G
    volumes:
      - neo4j-data:/data
      - neo4j-logs:/logs
      - neo4j-conf:/conf

  zookeeper:
    image: confluentinc/cp-zookeeper:latest
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
    volumes:
      - zookeeper-data:/var/lib/zookeeper/data
      - zookeeper-log:/var/lib/zookeeper/log

  kafka:
    image: confluentinc/cp-kafka:latest
    depends_on:
      - zookeeper
    ports:
      - "9092:9092"
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
    volumes:
      - kafka-data:/var/lib/kafka/data

volumes:
  neo4j-data:
  neo4j-logs:
  neo4j-conf:
  zookeeper-data:
  zookeeper-log:
  kafka-data:
```

### Building Docker Images

```bash
# Build all images
docker-compose build

# Build specific image
docker build -t agent-orchestration/mcp-server:latest -f dockerfiles/mcp-server.Dockerfile .
```

## Kubernetes Deployment

### Prerequisites

- Kubernetes cluster (1.19+)
- Helm (3.0+)
- kubectl configured to access your cluster

### Installing with Helm

1. Add the Helm repository:

```bash
helm repo add agent-orchestration https://helm.example.com/agent-orchestration
helm repo update
```

2. Create a values file (`values.yaml`):

```yaml
global:
  environment: production
  openaiApiKey: your-api-key

mcpServer:
  replicas: 3
  resources:
    requests:
      cpu: 2
      memory: 4Gi
    limits:
      cpu: 4
      memory: 8Gi

coreServices:
  replicas: 3
  resources:
    requests:
      cpu: 2
      memory: 4Gi
    limits:
      cpu: 4
      memory: 8Gi

neo4j:
  mode: cluster
  replicas: 3
  resources:
    requests:
      cpu: 4
      memory: 16Gi
    limits:
      cpu: 8
      memory: 32Gi

kafka:
  replicas: 3
  resources:
    requests:
      cpu: 2
      memory: 4Gi
    limits:
      cpu: 4
      memory: 8Gi
```

3. Install the Helm chart:

```bash
helm install agent-orchestration agent-orchestration/agent-orchestration -f values.yaml
```

### Kubernetes Manifest Examples

#### MCP Server Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mcp-server
  labels:
    app: mcp-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mcp-server
  template:
    metadata:
      labels:
        app: mcp-server
    spec:
      containers:
      - name: mcp-server
        image: agent-orchestration/mcp-server:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: openai-credentials
              key: api-key
        - name: NEO4J_URI
          value: "neo4j://neo4j-service:7687"
        - name: NEO4J_USERNAME
          valueFrom:
            secretKeyRef:
              name: neo4j-credentials
              key: username
        - name: NEO4J_PASSWORD
          valueFrom:
            secretKeyRef:
              name: neo4j-credentials
              key: password
        - name: KAFKA_BOOTSTRAP_SERVERS
          value: "kafka-service:9092"
        resources:
          requests:
            cpu: "2"
            memory: "4Gi"
          limits:
            cpu: "4"
            memory: "8Gi"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

#### MCP Server Service

```yaml
apiVersion: v1
kind: Service
metadata:
  name: mcp-server-service
spec:
  selector:
    app: mcp-server
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP
```

#### Ingress

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: agent-orchestration-ingress
  annotations:
    kubernetes.io/ingress.class: nginx
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  rules:
  - host: api.agent-orchestration.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: mcp-server-service
            port:
              number: 80
  tls:
  - hosts:
    - api.agent-orchestration.example.com
    secretName: agent-orchestration-tls
```

## Cloud Deployment

### AWS Setup

1. **Set up VPC and Subnets**

```bash
aws ec2 create-vpc --cidr-block 10.0.0.0/16 --tag-specifications 'ResourceType=vpc,Tags=[{Key=Name,Value=AgentOrchestrationVPC}]'
```

2. **Create EKS Cluster**

```bash
eksctl create cluster \
  --name agent-orchestration \
  --version 1.23 \
  --region us-west-2 \
  --nodegroup-name standard-nodes \
  --node-type t3.xlarge \
  --nodes 3 \
  --nodes-min 3 \
  --nodes-max 10 \
  --with-oidc \
  --managed
```

3. **Deploy Application using Helm**

```bash
helm install agent-orchestration agent-orchestration/agent-orchestration -f values.yaml
```

### GCP Setup

1. **Create GKE Cluster**

```bash
gcloud container clusters create agent-orchestration \
  --num-nodes=3 \
  --machine-type=e2-standard-4 \
  --region=us-central1
```

2. **Get Credentials**

```bash
gcloud container clusters get-credentials agent-orchestration --region=us-central1
```

3. **Deploy Application using Helm**

```bash
helm install agent-orchestration agent-orchestration/agent-orchestration -f values.yaml
```

### Azure Setup

1. **Create AKS Cluster**

```bash
az aks create \
  --resource-group myResourceGroup \
  --name agent-orchestration \
  --node-count 3 \
  --node-vm-size Standard_DS3_v2 \
  --enable-addons monitoring \
  --generate-ssh-keys
```

2. **Get Credentials**

```bash
az aks get-credentials --resource-group myResourceGroup --name agent-orchestration
```

3. **Deploy Application using Helm**

```bash
helm install agent-orchestration agent-orchestration/agent-orchestration -f values.yaml
```

## Configuration Management

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `OPENAI_API_KEY` | OpenAI API key | - | Yes |
| `NEO4J_URI` | Neo4j connection URI | `neo4j://localhost:7687` | Yes |
| `NEO4J_USERNAME` | Neo4j username | `neo4j` | Yes |
| `NEO4J_PASSWORD` | Neo4j password | - | Yes |
| `KAFKA_BOOTSTRAP_SERVERS` | Kafka bootstrap servers | `localhost:9092` | Yes |
| `MCP_SERVER_PORT` | MCP server port | `8000` | No |
| `LOG_LEVEL` | Logging level | `INFO` | No |
| `ENVIRONMENT` | Deployment environment | `development` | No |

### Configuration Files

Example `config.yaml`:

```yaml
server:
  port: 8000
  host: 0.0.0.0
  workers: 4
  timeout: 60

database:
  uri: neo4j://localhost:7687
  username: neo4j
  password: password
  pool_size: 10
  connection_timeout: 5000

kafka:
  bootstrap_servers: localhost:9092
  num_partitions: 3
  replication_factor: 1
  topics:
    - name: agent_events
      partitions: 4
      retention_ms: 604800000
    - name: evolution_events
      partitions: 4
      retention_ms: 604800000
    - name: feedback_events
      partitions: 4
      retention_ms: 604800000

openai:
  api_key: your-api-key
  default_model: gpt-4-turbo
  request_timeout: 30
```

## Monitoring and Observability

### Prometheus Integration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'agent-orchestration'
    scrape_interval: 5s
    static_configs:
      - targets: ['mcp-server:8000']
```

### Grafana Dashboard

Example dashboard configuration for agent metrics:

```json
{
  "annotations": {
    "list": [
      {
        "builtIn": 1,
        "datasource": "-- Grafana --",
        "enable": true,
        "hide": true,
        "iconColor": "rgba(0, 211, 255, 1)",
        "name": "Annotations & Alerts",
        "type": "dashboard"
      }
    ]
  },
  "editable": true,
  "gnetId": null,
  "graphTooltip": 0,
  "id": 1,
  "links": [],
  "panels": [
    {
      "aliasColors": {},
      "bars": false,
      "dashLength": 10,
      "dashes": false,
      "datasource": null,
      "fieldConfig": {
        "defaults": {},
        "overrides": []
      },
      "fill": 1,
      "fillGradient": 0,
      "gridPos": {
        "h": 8,
        "w": 12,
        "x": 0,
        "y": 0
      },
      "hiddenSeries": false,
      "id": 2,
      "legend": {
        "avg": false,
        "current": false,
        "max": false,
        "min": false,
        "show": true,
        "total": false,
        "values": false
      },
      "lines": true,
      "linewidth": 1,
      "nullPointMode": "null",
      "options": {
        "alertThreshold": true
      },
      "percentage": false,
      "pluginVersion": "7.5.5",
      "pointradius": 2,
      "points": false,
      "renderer": "flot",
      "seriesOverrides": [],
      "spaceLength": 10,
      "stack": false,
      "steppedLine": false,
      "targets": [
        {
          "expr": "sum(rate(agent_interactions_total[5m])) by (agent_id)",
          "interval": "",
          "legendFormat": "{{agent_id}}",
          "refId": "A"
        }
      ],
      "thresholds": [],
      "timeFrom": null,
      "timeRegions": [],
      "timeShift": null,
      "title": "Agent Interactions",
      "tooltip": {
        "shared": true,
        "sort": 0,
        "value_type": "individual"
      },
      "type": "graph",
      "xaxis": {
        "buckets": null,
        "mode": "time",
        "name": null,
        "show": true,
        "values": []
      },
      "yaxes": [
        {
          "format": "short",
          "label": null,
          "logBase": 1,
          "max": null,
          "min": null,
          "show": true
        },
        {
          "format": "short",
          "label": null,
          "logBase": 1,
          "max": null,
          "min": null,
          "show": true
        }
      ],
      "yaxis": {
        "align": false,
        "alignLevel": null
      }
    }
  ],
  "schemaVersion": 27,
  "style": "dark",
  "tags": [],
  "templating": {
    "list": []
  },
  "time": {
    "from": "now-6h",
    "to": "now"
  },
  "timepicker": {},
  "timezone": "",
  "title": "Agent Metrics",
  "uid": "agentMetrics",
  "version": 1
}
```

### Logging Configuration

```python
import logging
import json
from datetime import datetime

class JsonFormatter(logging.Formatter):
    """Format logs as JSON for structured logging."""
    
    def format(self, record):
        log_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        if hasattr(record, 'request_id'):
            log_record["request_id"] = record.request_id
            
        if hasattr(record, 'agent_id'):
            log_record["agent_id"] = record.agent_id
            
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_record)

def configure_logging(level=logging.INFO):
    """Configure structured JSON logging."""
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    
    logger = logging.getLogger("agent_orchestration")
    logger.setLevel(level)
    logger.addHandler(handler)
    
    return logger
```

## Scaling Strategies

### Horizontal Scaling

- **MCP Server**: Scale based on request rate
- **Core Services**: Scale based on event processing rate
- **Kafka**: Scale based on throughput and partition load
- **Neo4j**: Scale read replicas for read-heavy workloads

### Auto-scaling Configuration

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: mcp-server-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: mcp-server
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## Backup and Recovery

### Neo4j Backup

```bash
# Create Neo4j backup
neo4j-admin backup --backup-dir=/var/lib/neo4j/backups --database=neo4j

# Schedule regular backups
crontab -e
# Add the following line to run daily at 2 AM
0 2 * * * neo4j-admin backup --backup-dir=/var/lib/neo4j/backups --database=neo4j
```

### Kafka Backup

```bash
# Create Kafka topics backup
kafka-console-consumer --bootstrap-server localhost:9092 --topic agent_events --from-beginning > agent_events_backup.json

# Use Kafka Connect for continuous backup to cloud storage
```

## Security Hardening

### Network Security

- Implement network policies to restrict pod-to-pod communication
- Use TLS for all service communication
- Implement API gateway for external access

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: mcp-server-network-policy
spec:
  podSelector:
    matchLabels:
      app: mcp-server
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: api-gateway
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: neo4j
    ports:
    - protocol: TCP
      port: 7687
  - to:
    - podSelector:
        matchLabels:
          app: kafka
    ports:
    - protocol: TCP
      port: 9092
```

### Secret Management

- Use Kubernetes Secrets or external secret management (AWS Secrets Manager, HashiCorp Vault)
- Rotate credentials regularly
- Use service accounts with minimal permissions

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: openai-credentials
type: Opaque
data:
  api-key: <base64-encoded-api-key>
---
apiVersion: v1
kind: Secret
metadata:
  name: neo4j-credentials
type: Opaque
data:
  username: <base64-encoded-username>
  password: <base64-encoded-password>
```

## Troubleshooting Guide

### Common Issues

1. **Neo4j Connection Failures**
   - Check Neo4j is running: `kubectl get pods -l app=neo4j`
   - Verify connection parameters
   - Check Neo4j logs: `kubectl logs -l app=neo4j`

2. **Kafka Connection Issues**
   - Verify bootstrap servers configuration
   - Check Kafka broker status: `kubectl get pods -l app=kafka`
   - Check Zookeeper status: `kubectl get pods -l app=zookeeper`

3. **OpenAI API Errors**
   - Verify API key is valid
   - Check rate limits and quotas
   - Review OpenAI service status

### Debugging Tools

```bash
# Check pod status
kubectl get pods

# View pod logs
kubectl logs <pod-name>

# Describe pod details
kubectl describe pod <pod-name>

# Port forward for local debugging
kubectl port-forward <pod-name> 8000:8000

# Execute commands in container
kubectl exec -it <pod-name> -- /bin/bash
```

## Performance Tuning

### Neo4j Configuration

```
dbms.memory.heap.initial_size=2G
dbms.memory.heap.max_size=4G
dbms.memory.pagecache.size=2G
dbms.tx_state.memory_allocation=ON_HEAP
```

### Kafka Configuration

```
num.network.threads=3
num.io.threads=8
socket.send.buffer.bytes=102400
socket.receive.buffer.bytes=102400
socket.request.max.bytes=104857600
log.retention.hours=168
```

### JVM Tuning

```
-Xms4g
-Xmx4g
-XX:+UseG1GC
-XX:MaxGCPauseMillis=200
-XX:ParallelGCThreads=4
-XX:ConcGCThreads=4
-XX:InitiatingHeapOccupancyPercent=70
```
