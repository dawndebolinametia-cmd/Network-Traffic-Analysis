import pymysql

# Connect as root
conn = pymysql.connect(
    host='localhost',
    user='root',
    password='12345',
    port=3306
)
cursor = conn.cursor()

# Create database
cursor.execute("CREATE DATABASE IF NOT EXISTS our_mysql;")
print("Database 'our_mysql' created or already exists.")

# Create user
cursor.execute("CREATE USER IF NOT EXISTS 'debbie'@'localhost' IDENTIFIED BY '12345';")
print("User 'debbie' created or already exists.")

# Grant privileges
cursor.execute("GRANT ALL PRIVILEGES ON our_mysql.* TO 'debbie'@'localhost';")
print("Granted all privileges on 'our_mysql' to 'debbie'.")

# Flush privileges
cursor.execute("FLUSH PRIVILEGES;")
print("Privileges flushed.")

conn.commit()
conn.close()
print("Database and user setup complete.")
