import os
from sqlalchemy import create_engine

def connect_to_database():
    # Use environment variables with fallback defaults
    db_user = os.environ.get('DB_USER', 'ds_user')
    db_password = os.environ.get('DB_PASSWORD', 'userpass')
    db_host = os.environ.get('DB_HOST', '127.0.0.1')
    db_port = os.environ.get('DB_PORT', '3306')
    db_name = os.environ.get('DB_NAME', 'ds_project')
    
    engine = create_engine(f'mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}')
    return engine
