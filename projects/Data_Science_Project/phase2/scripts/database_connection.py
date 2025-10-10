from sqlalchemy import create_engine

def connect_to_database():
    db_user = 'ds_user'
    db_password = 'rootpass'
    db_host = 'db'
    db_port = '3306'
    db_name = 'ds_project'
    
    engine = create_engine(f'mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}')
    return engine
