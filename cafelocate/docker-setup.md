# Docker Setup for CaféLocate 🐳

This guide explains how to run the CaféLocate application using Docker and Docker Compose.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

## Services Overview

The `docker-compose.yml` defines the following services:

- **backend**: Django REST API server (port 8000)
- **db**: PostgreSQL database (port 5432)
- **redis**: Redis cache server (port 6379)
- **pgadmin**: PostgreSQL web admin interface (port 5050)
- **adminer**: Lightweight database management (port 8080) - dev only

## Quick Start

1. **Navigate to the project directory:**
   ```bash
   cd cafelocate
   ```

2. **Build and start all services:**
   ```bash
   docker-compose up --build
   ```

3. **Access the application:**
   - Django API: http://localhost:8000
   - PgAdmin: http://localhost:5050 (admin@cafelocate.com / admin123)
   - Adminer: http://localhost:8080

## Development Workflow

### First Time Setup
```bash
# Build the images
docker-compose build

# Run database migrations
docker-compose run --rm backend python manage.py migrate

# Create superuser (optional)
docker-compose run --rm backend python manage.py createsuperuser

# Start all services
docker-compose up
```

### Development Commands
```bash
# Start services in background
docker-compose up -d

# View logs
docker-compose logs -f backend

# Run Django management commands
docker-compose exec backend python manage.py shell
docker-compose exec backend python manage.py makemigrations

# Stop services
docker-compose down

# Rebuild after code changes
docker-compose up --build
```

### Database Management
- **PgAdmin**: Web interface at http://localhost:5050
  - Email: admin@cafelocate.com
  - Password: admin123
- **Adminer**: Lightweight interface at http://localhost:8080
- **Direct connection**: localhost:5432 with credentials from docker-compose.yml

## Environment Variables

Key environment variables (defined in docker-compose.yml):

```yaml
backend:
  DEBUG: True
  SECRET_KEY: django-insecure-dev-key-change-in-production
  DATABASE_URL: postgresql://cafelocate_user:cafelocate_password@db:5432/cafelocate_db
  CORS_ALLOWED_ORIGINS: http://localhost:5500,http://127.0.0.1:5500
```

## File Structure

```
cafelocate/
├── docker-compose.yml          # Main compose file
├── docker-compose.override.yml # Development overrides
├── backend/
│   ├── Dockerfile             # Django app container definition
│   ├── .dockerignore         # Files to exclude from build
│   ├── requirements.txt      # Python dependencies
│   └── ...                   # Django project files
├── data/                      # Mounted data volume
└── ml/                        # Mounted ML models volume
```

## Troubleshooting

### Common Issues

1. **Port conflicts**: Change ports in docker-compose.yml if 8000/5432/6379 are in use

2. **Database connection errors**: Wait for PostgreSQL to fully start (check logs with `docker-compose logs db`)

3. **Permission errors**: Ensure Docker has access to project files

4. **Build failures**: Clear Docker cache with `docker system prune`

### Useful Commands

```bash
# View running containers
docker-compose ps

# View service logs
docker-compose logs [service_name]

# Execute commands in running container
docker-compose exec backend bash

# Stop and remove everything
docker-compose down -v --remove-orphans

# Rebuild specific service
docker-compose build backend
```

## Production Deployment

For production deployment:

1. **Update environment variables** in docker-compose.yml:
   - Set `DEBUG=False`
   - Use strong `SECRET_KEY`
   - Configure proper `ALLOWED_HOSTS`

2. **Use environment-specific compose files**:
   ```bash
   docker-compose -f docker-compose.yml -f docker-compose.prod.yml up
   ```

3. **Add reverse proxy** (nginx) for static files and SSL

4. **Configure proper volumes** for data persistence

5. **Set up monitoring** and logging

## Volumes

The setup uses named volumes for data persistence:
- `postgres_data`: Database files
- `redis_data`: Redis cache
- `pgadmin_data`: PgAdmin configuration

Data is preserved between container restarts but can be removed with:
```bash
docker-compose down -v
```