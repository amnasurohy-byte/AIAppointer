# Deployment Guide - AI Appointer Assist

## Table of Contents
1. [On-Premise Air-Gapped Deployment](#on-premise-air-gapped-deployment)
2. [Cloud/Server Deployment](#cloudserver-deployment)
3. [Docker Deployment](#docker-deployment)
4. [Configuration](#configuration)
5. [Troubleshooting](#troubleshooting)

---

## On-Premise Air-Gapped Deployment

### Prerequisites
- Linux/Windows server with Python 3.8+
- No internet access required
- Minimum 4GB RAM, 10GB disk space

### Step 1: Prepare Deployment Package (On Internet-Connected Machine)

```bash
# Download all dependencies
pip download -r requirements.txt -d wheels/

# Create deployment archive
tar -czf ai-appointer-deploy.tar.gz \
    AIAppointer/ \
    wheels/ \
    scripts/install.sh \
    scripts/start.sh \
    scripts/stop.sh
```

### Step 2: Transfer to Air-Gapped Server
- USB drive
- Secure file transfer
- Physical media

### Step 3: Install on Air-Gapped Server

```bash
# Extract package
tar -xzf ai-appointer-deploy.tar.gz
cd AIAppointer

# Run installation script
chmod +x scripts/install.sh
./scripts/install.sh
```

### Step 4: Configure

```bash
# Copy environment template
cp .env.example .env

# Edit configuration
nano .env
```

### Step 5: Start Application

```bash
# Start the application
chmod +x scripts/start.sh
./scripts/start.sh

# Application will be available at http://localhost:8501
```

### Step 6: Stop Application

```bash
chmod +x scripts/stop.sh
./scripts/stop.sh
```

---

## Cloud/Server Deployment

### Option 1: Direct Deployment

```bash
# Clone repository
git clone <repository-url>
cd AIAppointer

# Install dependencies
pip install -r requirements.txt

# Configure
cp .env.example .env
nano .env

# Train model (if needed)
python scripts/train_model.py

# Start application
streamlit run src/app.py --server.port 8501
```

### Option 2: Systemd Service (Linux)

Create `/etc/systemd/system/ai-appointer.service`:

```ini
[Unit]
Description=AI Appointer Assist
After=network.target

[Service]
Type=simple
User=appointer
WorkingDirectory=/opt/AIAppointer
Environment="PATH=/opt/AIAppointer/venv/bin"
ExecStart=/opt/AIAppointer/venv/bin/streamlit run src/app.py --server.port 8501
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable ai-appointer
sudo systemctl start ai-appointer
sudo systemctl status ai-appointer
```

---

## Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/ ./src/
COPY models/ ./models/
COPY data/ ./data/
COPY config.py .

# Expose port
EXPOSE 8501

# Run application
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Build and Run

```bash
# Build image
docker build -t ai-appointer:latest .

# Run container
docker run -d \
  -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  --name ai-appointer \
  ai-appointer:latest
```

### Docker Compose

```yaml
version: '3.8'

services:
  ai-appointer:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    environment:
      - LOG_LEVEL=INFO
    restart: unless-stopped
```

Run with:
```bash
docker-compose up -d
```

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `APPOINTER_DATA_DIR` | Data directory path | `./data` |
| `DATASET_FILE` | Dataset filename | `hr_data.csv` |
| `APPOINTER_MODELS_DIR` | Models directory | `./models` |
| `DEFAULT_RANK_FLEX` | Default rank flexibility | `0` |
| `UI_TITLE` | Application title | `AI Appointer Assist` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `STREAMLIT_SERVER_PORT` | Server port | `8501` |
| `STREAMLIT_SERVER_ADDRESS` | Server address | `0.0.0.0` |

### config.py

Edit `config.py` for advanced configuration:
- Dataset paths
- Model parameters
- UI customization
- Feature engineering settings

---

## Security Considerations

### Production Checklist
- [ ] Change default ports
- [ ] Enable HTTPS (use reverse proxy like Nginx)
- [ ] Set up authentication (Streamlit supports basic auth)
- [ ] Configure firewall rules
- [ ] Enable audit logging
- [ ] Regular backups of models and data
- [ ] Keep dependencies updated

### HTTPS Setup (Nginx)

```nginx
server {
    listen 443 ssl;
    server_name appointer.yourdomain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

---

## Troubleshooting

### Application Won't Start

**Check Python version**:
```bash
python --version  # Should be 3.8+
```

**Check dependencies**:
```bash
pip list | grep streamlit
```

**Check logs**:
```bash
tail -f logs/appointer.log
```

### Model Not Found

**Verify models directory**:
```bash
ls -la models/
# Should contain: lgbm_model.pkl, all_constraints.json, etc.
```

**Retrain if needed**:
```bash
python scripts/train_model.py
```

### Port Already in Use

**Change port in .env**:
```bash
STREAMLIT_SERVER_PORT=8502
```

Or kill existing process:
```bash
# Linux
lsof -ti:8501 | xargs kill -9

# Windows
netstat -ano | findstr :8501
taskkill /PID <PID> /F
```

### Performance Issues

**Increase resources**:
- RAM: Minimum 4GB recommended
- CPU: 2+ cores recommended

**Optimize dataset**:
- Reduce dataset size if too large
- Use sampling for development

---

## Monitoring

### Health Check Endpoint

Streamlit provides health check at:
```
http://localhost:8501/healthz
```

### Logs

Application logs location:
- Development: Console output
- Production: `logs/appointer.log`

Enable verbose logging:
```bash
export LOG_LEVEL=DEBUG
```

---

## Backup and Recovery

### Backup

```bash
# Backup models
tar -czf models-backup-$(date +%Y%m%d).tar.gz models/

# Backup data
tar -czf data-backup-$(date +%Y%m%d).tar.gz data/
```

### Restore

```bash
# Restore models
tar -xzf models-backup-YYYYMMDD.tar.gz

# Restore data
tar -xzf data-backup-YYYYMMDD.tar.gz
```

---

## Upgrading

### From Git

```bash
# Backup current version
cp -r AIAppointer AIAppointer.backup

# Pull updates
cd AIAppointer
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade

# Restart application
./scripts/stop.sh
./scripts/start.sh
```

### Manual Upgrade

1. Backup current installation
2. Download new version
3. Copy `models/` and `data/` from backup
4. Update configuration
5. Restart application

---

## Support

For issues and questions:
- Check logs: `logs/appointer.log`
- Review documentation: `docs/`
- GitHub Issues: [repository-url]/issues
