import sqlite3
from initiation import Initiation

class FunctionCalling(Initiation):
    def __init__(self):
        super().__init__()
        self.conn = sqlite3.connect("../../input_database/alerting_db")
        print("Opened database successfully")

    def get_table_names(self, conn):
        """Return a list of table names."""
        table_names = []
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
        for table in tables.fetchall():
            table_names.append(table[0])
        return table_names

    def get_column_names(self, conn, table_name):
        """Return a list of column names."""
        column_names = []
        columns = conn.execute(f"PRAGMA table_info('{table_name}');").fetchall()
        for col in columns:
            column_names.append(col[1])
        return column_names

    def get_database_info(self, conn):
        """Return a list of dicts containing the table name and columns for each table in the database."""
        table_dicts = []
        for table_name in self.get_table_names(conn):
            columns_names = self.get_column_names(conn, table_name)
            table_dicts.append({"table_name": table_name, "column_names": columns_names})
        return table_dicts
    
    def database_schema_string(self):
        database_schema_dict = self.get_database_info(self.conn)
        database_schema_string = "\n".join(
            [
                f"Table: {table['table_name']}\nColumns: {', '.join(table['column_names'])}"
                for table in database_schema_dict
            ]
        )
        return database_schema_string

    def ask_database(self, query):
        """Function to query SQLite database with a provided SQL query."""
        try:
            results = str(self.conn.execute(query).fetchall())
        except Exception as e:
            results = f"query failed with error: {e}"
        return results
