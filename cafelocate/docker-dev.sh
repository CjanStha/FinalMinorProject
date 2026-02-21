#!/bin/bash
# CaféLocate Docker Development Script for Linux/Mac
# Usage: ./docker-dev.sh [command]

set -e

case "$1" in
    build)
        echo "Building Docker images..."
        docker-compose build
        ;;

    up)
        echo "Starting all services..."
        docker-compose up -d
        echo
        echo "Services started! Access points:"
        echo "- Django API: http://localhost:8000"
        echo "- PgAdmin: http://localhost:5050 (admin@cafelocate.com / admin123)"
        echo "- Adminer: http://localhost:8080"
        ;;

    down)
        echo "Stopping all services..."
        docker-compose down
        ;;

    logs)
        if [ -n "$2" ]; then
            docker-compose logs -f "$2"
        else
            docker-compose logs -f
        fi
        ;;

    shell)
        echo "Opening Django shell..."
        docker-compose exec backend python manage.py shell
        ;;

    migrate)
        echo "Running database migrations..."
        docker-compose exec backend python manage.py migrate
        ;;

    makemigrations)
        echo "Creating new migrations..."
        docker-compose exec backend python manage.py makemigrations
        ;;

    createsuperuser)
        echo "Creating Django superuser..."
        docker-compose exec backend python manage.py createsuperuser
        ;;

    restart)
        echo "Restarting backend service..."
        docker-compose restart backend
        ;;

    clean)
        echo "Cleaning up Docker resources..."
        docker-compose down -v --remove-orphans
        docker system prune -f
        ;;

    *)
        echo "CaféLocate Docker Development Helper"
        echo
        echo "Usage: $0 [command]"
        echo
        echo "Commands:"
        echo "  build           - Build Docker images"
        echo "  up              - Start all services"
        echo "  down            - Stop all services"
        echo "  logs [service]  - View logs (optionally for specific service)"
        echo "  shell           - Open Django shell"
        echo "  migrate         - Run database migrations"
        echo "  makemigrations  - Create new migrations"
        echo "  createsuperuser - Create Django superuser"
        echo "  restart         - Restart backend service"
        echo "  clean           - Clean up Docker resources"
        echo
        echo "Examples:"
        echo "  $0 up"
        echo "  $0 logs backend"
        echo "  $0 shell"
        ;;
esac