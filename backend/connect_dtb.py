import pyodbc

DRIVER_NAME = 'SQL Server'
SERVER_NAME = r'LAPTOP-I7G4TV8E\SQLEXPRESS'
DATABASE_NAME = 'defaced'

connection_string = (
    f"DRIVER={{{DRIVER_NAME}}};"
    f"SERVER={SERVER_NAME};"
    f"DATABASE={DATABASE_NAME};"
    f"Trusted_Connection=yes;"
)

conn = pyodbc.connect(connection_string)
print("Kết nối thành công:", conn)

def get_connection():
    return pyodbc.connect(connection_string)
