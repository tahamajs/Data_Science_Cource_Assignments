#!/bin/bash
# Docker entrypoint script for DS Project
# This script ensures proper initialization and execution of the data pipeline

set -e  # Exit immediately if a command exits with a non-zero status

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to wait for MySQL to be ready
wait_for_mysql() {
    log_info "Waiting for MySQL to be ready..."
    
    local max_attempts=60
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if mysqladmin ping -h "$DB_HOST" -u "$DB_USER" -p"$DB_PASSWORD" --silent 2>/dev/null; then
            log_success "MySQL is ready!"
            return 0
        fi
        
        log_info "Attempt $attempt/$max_attempts: MySQL not ready yet, waiting..."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    log_error "MySQL failed to become ready after $max_attempts attempts"
    return 1
}

# Function to check database connection
check_database() {
    log_info "Checking database connection..."
    
    if mysql -h "$DB_HOST" -u "$DB_USER" -p"$DB_PASSWORD" "$DB_NAME" -e "SELECT 1;" >/dev/null 2>&1; then
        log_success "Database connection successful!"
        return 0
    else
        log_error "Failed to connect to database"
        return 1
    fi
}

# Function to seed database
seed_database() {
    log_info "Checking if database needs seeding..."
    
    # Check if tables have data
    local trip_count=$(mysql -h "$DB_HOST" -u "$DB_USER" -p"$DB_PASSWORD" "$DB_NAME" -se "SELECT COUNT(*) FROM uber_trips;" 2>/dev/null || echo "0")
    
    if [ "$trip_count" -eq "0" ]; then
        log_info "Database is empty. Seeding data..."
        python scripts/seed_database.py
        
        if [ $? -eq 0 ]; then
            log_success "Database seeded successfully!"
        else
            log_error "Database seeding failed!"
            return 1
        fi
    else
        log_success "Database already contains data (trips: $trip_count). Skipping seeding."
    fi
}

# Function to run the pipeline
run_pipeline() {
    log_info "Starting data processing pipeline..."
    
    local start_time=$(date +%s)
    
    python pipeline.py
    local exit_code=$?
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    if [ $exit_code -eq 0 ]; then
        log_success "Pipeline completed successfully in ${duration} seconds!"
        return 0
    else
        log_error "Pipeline failed with exit code $exit_code"
        return 1
    fi
}

# Function to display system information
display_system_info() {
    log_info "System Information:"
    echo "  Python Version: $(python --version)"
    echo "  Working Directory: $(pwd)"
    echo "  Database Host: $DB_HOST"
    echo "  Database Name: $DB_NAME"
    echo "  Environment: ${ENVIRONMENT:-production}"
    echo ""
}

# Function to cleanup on exit
cleanup() {
    log_info "Cleaning up..."
    # Add any cleanup tasks here
    log_success "Cleanup completed"
}

# Register cleanup function to run on exit
trap cleanup EXIT

# Main execution
main() {
    echo "=================================================="
    echo "   Data Science Project - Pipeline Execution"
    echo "=================================================="
    echo ""
    
    # Display system information
    display_system_info
    
    # Wait for MySQL to be ready
    if ! wait_for_mysql; then
        log_error "Cannot proceed without database connection"
        exit 1
    fi
    
    # Check database connection
    if ! check_database; then
        log_error "Database connection check failed"
        exit 1
    fi
    
    # Seed database if needed
    if ! seed_database; then
        log_error "Database seeding failed"
        exit 1
    fi
    
    # Run the pipeline
    if ! run_pipeline; then
        log_error "Pipeline execution failed"
        exit 1
    fi
    
    echo ""
    echo "=================================================="
    log_success "All tasks completed successfully! ✅"
    echo "=================================================="
    
    # Keep container running if in development mode
    if [ "${ENVIRONMENT}" = "development" ]; then
        log_info "Running in development mode. Container will stay alive."
        tail -f /dev/null
    fi
}

# Run main function
main "$@"
