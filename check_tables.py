from sqlalchemy import create_engine, text
from config import DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, DB_PORT

engine = create_engine(f'mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')
with engine.connect() as conn:
    result = conn.execute(text('SHOW TABLES'))
    tables = [row[0] for row in result]
    print('Tables:', tables)
    for table in ['network_traffic', 'anomaly_results', 'supervised_predictions', 'summary_stats']:
        if table in tables:
            result = conn.execute(text(f'SELECT COUNT(*) FROM {table}'))
            count = result.fetchone()[0]
            print(f'{table}: {count} rows')
        else:
            print(f'{table}: Table does not exist')
