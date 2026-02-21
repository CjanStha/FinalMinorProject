@echo off
REM CaféLocate Docker Development Script for Windows
REM Usage: docker-dev.bat [command]

if "%1"=="build" (
    echo Building Docker images...
    docker-compose build
    goto end
)

if "%1"=="up" (
    echo Starting all services...
    docker-compose up -d
    echo.
    echo Services started! Access points:
    echo - Django API: http://localhost:8000
    echo - PgAdmin: http://localhost:5050 (admin@cafelocate.com / admin123)
    echo - Adminer: http://localhost:8080
    goto end
)

if "%1"=="down" (
    echo Stopping all services...
    docker-compose down
    goto end
)

if "%1"=="logs" (
    if "%2"=="" (
        docker-compose logs -f
    ) else (
        docker-compose logs -f %2
    )
    goto end
)

if "%1"=="shell" (
    echo Opening Django shell...
    docker-compose exec backend python manage.py shell
    goto end
)

if "%1"=="migrate" (
    echo Running database migrations...
    docker-compose exec backend python manage.py migrate
    goto end
)

if "%1"=="makemigrations" (
    echo Creating new migrations...
    docker-compose exec backend python manage.py makemigrations
    goto end
)

if "%1"=="createsuperuser" (
    echo Creating Django superuser...
    docker-compose exec backend python manage.py createsuperuser
    goto end
)

if "%1"=="restart" (
    echo Restarting backend service...
    docker-compose restart backend
    goto end
)

if "%1"=="clean" (
    echo Cleaning up Docker resources...
    docker-compose down -v --remove-orphans
    docker system prune -f
    goto end
)

echo CaféLocate Docker Development Helper
echo.
echo Usage: docker-dev.bat [command]
echo.
echo Commands:
echo   build           - Build Docker images
echo   up              - Start all services
echo   down            - Stop all services
echo   logs [service]  - View logs (optionally for specific service)
echo   shell           - Open Django shell
echo   migrate         - Run database migrations
echo   makemigrations  - Create new migrations
echo   createsuperuser - Create Django superuser
echo   restart         - Restart backend service
echo   clean           - Clean up Docker resources
echo.
echo Examples:
echo   docker-dev.bat up
echo   docker-dev.bat logs backend
echo   docker-dev.bat shell

:end