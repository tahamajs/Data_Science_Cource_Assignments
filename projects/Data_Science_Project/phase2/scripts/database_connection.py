"""
ماژول اتصال به دیتابیس
این ماژول مسئول ایجاد اتصال به MySQL با استفاده از SQLAlchemy است
"""

import os
from sqlalchemy import create_engine


def connect_to_database():
    """
    ایجاد اتصال به دیتابیس MySQL

    Returns:
        engine: SQLAlchemy engine object
    """
    db_user = os.getenv("DB_USER", "ds_user")
    db_password = os.getenv("DB_PASS", "userpass")
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "3306")
    db_name = os.getenv("DB_NAME", "ds_project")

    connection_string = (
        f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    )
    engine = create_engine(connection_string, pool_pre_ping=True)

    return engine
