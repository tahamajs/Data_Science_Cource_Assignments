# scripts/seed_database.py

import os
import sys
import pandas as pd
from sqlalchemy import create_engine, text

def seed_table(engine, table_name, file_path, chunksize=None):
    """
    Seeds a single table, but only if it's empty.
    Handles both small files and large files in chunks.
    """
    try:
        # First, check if the table already has data
        with engine.connect() as connection:
            result = connection.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
            count = result.scalar()

        # If the table is not empty, skip it.
        if count > 0:
            print(f"✅ Table '{table_name}' is not empty. Skipping seeding.")
            return

        # If the table is empty, proceed with seeding.
        print(f"🔄 Table '{table_name}' is empty. Seeding data from '{file_path}'...")

        # If no chunksize is provided, load the whole file at once (for small files)
        if chunksize is None:
            df = pd.read_csv(file_path)
            df.to_sql(table_name, engine, if_exists='append', index=False)
            print(f"✅ Data for '{table_name}' loaded successfully.")
        
        # If chunksize is provided, load the file in chunks (for large files)
        else:
            chunk_iterator = pd.read_csv(file_path, chunksize=chunksize)
            chunk_num = 1
            for chunk in chunk_iterator:
                # You can add column renaming logic here if needed for specific tables
                # if table_name == 'uber_trips':
                #     chunk.rename(columns={'locationID': 'pickup_location_id'}, inplace=True)

                print(f"  -> Writing chunk {chunk_num} to '{table_name}'...")
                chunk.to_sql(table_name, engine, if_exists='append', index=False)
                print(f"  ✅ Chunk {chunk_num} loaded successfully.")
                chunk_num += 1
            print(f"✅ All chunks for '{table_name}' loaded successfully.")

    except Exception as e:
        print(f"❌ Error seeding table '{table_name}': {e}")
        # Re-raise the exception to stop the script if something goes wrong
        raise e

def main():
    """
    Connects to the database and seeds all necessary tables idempotently.
    """
    db_host = os.environ.get('DB_HOST')
    db_port = os.environ.get('DB_PORT')
    db_user = os.environ.get('DB_USER')
    db_password = os.environ.get('DB_PASSWORD')
    db_name = os.environ.get('DB_NAME')

    if not all([db_host, db_port, db_user, db_password, db_name]):
        print("❌ Error: Missing one or more database environment variables.")
        sys.exit(1)

    engine = None
    try:
        db_url = f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        engine = create_engine(db_url)
        print("✅ Successfully connected to the database.")

        # --- Seed all tables using the robust function ---
        seed_table(engine, 'taxi_zones', 'database/taxi_zone_lookup_coordinates.csv')
        seed_table(engine, 'weather_data', 'database/weather_data_cleaned.csv')
        seed_table(engine, 'uber_trips', 'database/uber_trips_processed.csv', chunksize=50000)

        print("\n🎉 Database seeding process completed.")

    except Exception as e:
        print(f"\n❌ A critical error occurred during the seeding process: {e}")
        sys.exit(1)
    
    finally:
        # Ensure the database connection is always closed
        if engine:
            engine.dispose()
            print("🔌 Database connection closed.")


if __name__ == "__main__":
    main()
