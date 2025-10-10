# Docker Setup Guide

This directory contains Docker configuration files for the Data Science Project.

## 📁 Directory Structure

```
docker/
├── mysql/
│   └── my.cnf          # MySQL optimization configuration
├── jupyter/
│   └── Dockerfile      # Jupyter Lab container
└── README.md           # This file
```

## 🐳 Docker Services

### 1. MySQL Database (`db`)

**Image**: `mysql:8.0`

**Purpose**: Stores Uber trips, weather data, and taxi zone information

**Configuration**:

- **Port**: 3307 (host) → 3306 (container)
- **Database**: `ds_project`
- **User**: `ds_user`
- **Password**: `userpass` (change in production!)
- **Root Password**: `rootpass`

**Volumes**:

- `mysql_data`: Persistent database storage
- `./database/schema.sql`: Auto-import on first run
- `./docker/mysql/my.cnf`: Custom MySQL configuration

**Health Check**: Pings MySQL every 10s to ensure readiness

**Optimization** (`my.cnf`):

- Buffer pool: 1GB
- Max connections: 1000
- Optimized for large data operations

### 2. Python Application (`app`)

**Build**: Custom Dockerfile (multi-stage build)

**Purpose**: Runs data pipeline and processing scripts

**Features**:

- Multi-stage build for smaller image size
- Health checks for monitoring
- Automatic database seeding
- Pipeline execution on startup

**Environment Variables**:

```bash
DB_HOST=db
DB_PORT=3306
DB_USER=ds_user
DB_PASSWORD=userpass
DB_NAME=ds_project
PYTHONUNBUFFERED=1
```

**Volumes**:

- `./database`: Data files (read-only)
- `./models`: Trained models (read-write)
- `./output`: Pipeline outputs (read-write)
- `./logs`: Execution logs (read-write)
- `./visualizations`: Generated plots (read-write)

**Depends On**: MySQL service (with health check)

### 3. phpMyAdmin (`phpmyadmin`) - Optional

**Image**: `phpmyadmin:latest`

**Purpose**: Web-based database management interface

**Access**: http://localhost:8080

**Profile**: `tools` (start with `--profile tools`)

**Login**:

- Server: `db`
- Username: `root`
- Password: `rootpass`

**Features**:

- Execute SQL queries
- Browse tables
- Export/import data
- Visual query builder

### 4. Jupyter Lab (`jupyter`) - Optional

**Build**: `docker/jupyter/Dockerfile`

**Purpose**: Interactive data analysis and experimentation

**Access**: http://localhost:8888

**Profile**: `development` (start with `--profile development`)

**Token**: Set via `JUPYTER_TOKEN` env var

**Features**:

- JupyterLab interface
- Pre-installed data science libraries
- Access to all project files
- Database connectivity

**Volumes**:

- `./notebooks`: Jupyter notebooks
- `./database`: Data files
- `./models`: Model files
- `./visualizations`: Outputs

## 🚀 Usage

### Basic Usage

```bash
# Start core services (MySQL + App)
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f
```

### With Optional Services

```bash
# Start with phpMyAdmin
docker-compose --profile tools up -d

# Start with Jupyter
docker-compose --profile development up -d

# Start with all services
docker-compose --profile tools --profile development up -d
```

### Common Operations

```bash
# Rebuild services
docker-compose up -d --build

# Restart specific service
docker-compose restart app

# Execute command in container
docker-compose exec app python scripts/load_data.py

# Open bash shell in app container
docker-compose exec app bash

# Open MySQL shell
docker-compose exec db mysql -u ds_user -p

# View real-time logs
docker-compose logs -f app
```

## 🔧 Customization

### MySQL Configuration

Edit `docker/mysql/my.cnf` to customize MySQL settings:

```ini
[mysqld]
# Increase buffer pool for better performance
innodb_buffer_pool_size=2G

# Increase max connections
max_connections=2000

# Enable slow query log
slow_query_log=1
long_query_time=1
```

Then restart:

```bash
docker-compose restart db
```

### Application Environment

Create or edit `.env` file:

```bash
# Database
DB_HOST=db
DB_PORT=3306
DB_USER=ds_user
DB_PASSWORD=your_secure_password
DB_NAME=ds_project

# Application
ENVIRONMENT=production
LOG_LEVEL=INFO

# Performance
CHUNK_SIZE=50000
MAX_WORKERS=4
```

### Docker Compose Override

Create `docker-compose.override.yml` for local customizations:

```yaml
version: "3.8"

services:
  app:
    environment:
      - DEBUG=true
    volumes:
      - ./custom_scripts:/app/custom_scripts

  db:
    ports:
      - "3306:3306" # Use standard port locally
```

## 📊 Monitoring

### Health Checks

Check service health:

```bash
docker-compose ps
```

Output shows health status:

```
Name                State          Ports
─────────────────────────────────────────────────────
ds_project_mysql   Up (healthy)   3306/tcp
ds_project_app     Up
```

### Logs

```bash
# All services
docker-compose logs

# Specific service
docker-compose logs app

# Follow logs
docker-compose logs -f

# Last 100 lines
docker-compose logs --tail=100

# Since timestamp
docker-compose logs --since 2025-10-10T10:00:00
```

### Resource Usage

```bash
# Container stats
docker stats

# Detailed inspection
docker-compose exec app top

# Disk usage
docker system df
```

## 🔐 Security Considerations

### Production Deployment

1. **Change Default Passwords**:

   ```bash
   # In .env file
   MYSQL_ROOT_PASSWORD=strong_random_password
   DB_PASSWORD=another_strong_password
   ```

2. **Use Secrets** (Docker Swarm/Kubernetes):

   ```yaml
   services:
     db:
       environment:
         MYSQL_ROOT_PASSWORD_FILE: /run/secrets/mysql_root_password
       secrets:
         - mysql_root_password

   secrets:
     mysql_root_password:
       external: true
   ```

3. **Disable phpMyAdmin in Production**:

   - Don't use `--profile tools` in production
   - Or restrict access with firewall rules

4. **Use Read-Only Mounts**:

   ```yaml
   volumes:
     - ./database:/app/database:ro # Read-only
   ```

5. **Network Isolation**:

   ```yaml
   networks:
     frontend:
       internal: false
     backend:
       internal: true # No external access
   ```

6. **Regular Updates**:
   ```bash
   docker-compose pull
   docker-compose up -d --build
   ```

## 🐛 Troubleshooting

### MySQL Won't Start

**Issue**: Container exits immediately

**Check**:

```bash
docker-compose logs db
```

**Common Causes**:

- Port 3307 already in use
- Insufficient memory
- Corrupted data volume

**Solutions**:

```bash
# Change port in docker-compose.yml
ports:
  - "3308:3306"

# Remove data volume (CAUTION: deletes all data)
docker-compose down -v
docker-compose up -d
```

### Application Can't Connect to Database

**Issue**: `Can't connect to MySQL server on 'db'`

**Check**:

1. Is MySQL healthy?

   ```bash
   docker-compose ps db
   ```

2. Are they on the same network?
   ```bash
   docker network inspect ds_project_network
   ```

**Solution**:

```bash
# Restart services in correct order
docker-compose down
docker-compose up -d db
# Wait for db to be healthy
docker-compose up -d app
```

### Out of Memory

**Issue**: Container killed unexpectedly

**Check**:

```bash
docker stats
```

**Solution**:

- Increase Docker memory limit (Docker Desktop → Settings → Resources)
- Reduce `innodb_buffer_pool_size` in `my.cnf`
- Use smaller `CHUNK_SIZE` in application

### Permission Issues

**Issue**: `Permission denied` errors

**Solution**:

```bash
# Fix permissions
chmod -R 755 docker-entrypoint.sh
chmod -R 777 output/ logs/ visualizations/

# Or run as specific user
docker-compose run --user $(id -u):$(id -g) app python pipeline.py
```

## 📈 Performance Tuning

### MySQL Optimization

For datasets > 10M records:

```ini
# docker/mysql/my.cnf
[mysqld]
innodb_buffer_pool_size=2G          # 50-70% of available RAM
innodb_log_file_size=512M           # Larger for bulk inserts
innodb_flush_log_at_trx_commit=2    # Faster, less durable
max_allowed_packet=512M             # For large queries
```

### Application Optimization

```yaml
# docker-compose.yml
services:
  app:
    deploy:
      resources:
        limits:
          cpus: "4"
          memory: 8G
        reservations:
          cpus: "2"
          memory: 4G
```

### Network Performance

```yaml
networks:
  ds_network:
    driver: bridge
    driver_opts:
      com.docker.network.driver.mtu: 1500
```

## 🔄 CI/CD Integration

### GitHub Actions

```yaml
# .github/workflows/docker.yml
name: Docker Build

on: [push]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Build Docker image
        run: docker-compose build app

      - name: Run tests in container
        run: docker-compose run app pytest tests/
```

### Docker Hub Integration

```bash
# Build and tag
docker build -t username/ds-project:latest .

# Push to Docker Hub
docker push username/ds-project:latest

# Use in docker-compose.yml
services:
  app:
    image: username/ds-project:latest
```

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [MySQL Docker Documentation](https://hub.docker.com/_/mysql)
- [Best Practices for Writing Dockerfiles](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)

## 🤝 Contributing

When modifying Docker configuration:

1. Test changes locally
2. Document any new environment variables
3. Update this README
4. Test on clean environment
5. Submit pull request

## 📄 License

Part of the Data Science Project - Academic Use
