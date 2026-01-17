# Secrets Management Strategy

## Overview

This document outlines approaches for managing secrets (API keys, passwords, tokens) in Nova Aetus.

## Current Implementation

Currently using `.env` file with `.gitignore` protection. This is acceptable for development but needs enhancement for production.

## Recommended Approaches

### 1. Environment Variables (Current - Development Only)

**Pros**: Simple, no additional infrastructure
**Cons**: Not suitable for production, manual management

**Current Setup**:
```bash
# .env file (gitignored)
DISCORD_WEBHOOK_URL=...
TIMESCALE_PASSWORD=...
ALPACA_API_KEY=...
```

### 2. HashiCorp Vault (Recommended for Production)

#### Setup
```bash
# Install Vault (example)
wget https://releases.hashicorp.com/vault/1.15.0/vault_1.15.0_linux_amd64.zip
unzip vault_1.15.0_linux_amd64.zip
sudo mv vault /usr/local/bin/
```

#### Configuration
```hcl
# vault.hcl
storage "file" {
  path = "/opt/vault/data"
}

listener "tcp" {
  address = "127.0.0.1:8200"
  tls_disable = 1
}
```

#### Integration Example
```python
# src/nova/core/secrets.py (to be created)
import hvac

class VaultSecrets:
    def __init__(self, vault_url: str, vault_token: str):
        self.client = hvac.Client(url=vault_url, token=vault_token)

    def get_secret(self, path: str, key: str) -> str:
        response = self.client.secrets.kv.v2.read_secret_version(path=path)
        return response['data']['data'][key]
```

#### Usage in Application
```python
# Load from Vault instead of .env
vault = VaultSecrets(vault_url="http://localhost:8200", vault_token=os.getenv("VAULT_TOKEN"))
discord_webhook = vault.get_secret("secret/nova-aetus", "discord_webhook_url")
```

### 3. AWS Secrets Manager

#### Setup
```bash
pip install boto3
```

#### Integration
```python
import boto3
import json

def get_secret(secret_name: str) -> dict:
    client = boto3.client('secretsmanager')
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response['SecretString'])
```

### 4. Docker Secrets (Docker Swarm)

```yaml
# docker-compose.yml
services:
  nova_aetus:
    secrets:
      - discord_webhook_url
      - db_password

secrets:
  discord_webhook_url:
    external: true
  db_password:
    file: ./secrets/db_password.txt
```

### 5. Kubernetes Secrets

```yaml
# secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: nova-aetus-secrets
type: Opaque
stringData:
  discord-webhook-url: "..."
  db-password: "..."
```

```python
# Load from Kubernetes
import os
webhook_url = os.getenv('DISCORD_WEBHOOK_URL')  # Injected by K8s
```

## Best Practices

### 1. Never Commit Secrets
- ✅ `.env` in `.gitignore`
- ✅ Use `.env.example` with placeholder values
- ✅ Scan codebase for hardcoded secrets

### 2. Rotate Secrets Regularly
- Set up rotation schedule (e.g., quarterly)
- Update secrets in secure storage
- Restart application to load new secrets

### 3. Least Privilege
- Use separate API keys per environment
- Limit API key permissions
- Use read-only keys where possible

### 4. Audit Access
- Log secret access (if using Vault)
- Monitor for unusual access patterns
- Alert on unauthorized access attempts

## Migration Path

### Phase 1: Current (Development)
- Use `.env` file
- Keep `.gitignore` updated
- Document secrets in `.env.example`

### Phase 2: Transition
- Support both `.env` and Vault
- Fallback to `.env` if Vault unavailable
- Gradual migration

### Phase 3: Production
- Full Vault integration
- Remove `.env` file dependency
- Automated secret rotation

## Implementation Priority

1. **Immediate**: Document current `.env` approach
2. **Short-term**: Add Vault support (optional, with fallback)
3. **Long-term**: Full secrets manager integration

## Security Checklist

- [ ] All secrets in `.gitignore`
- [ ] `.env.example` with placeholders
- [ ] No hardcoded secrets in code
- [ ] Secrets rotation process documented
- [ ] Access logs for secret retrieval
- [ ] Separate secrets per environment
- [ ] Encrypted backup of secrets store
