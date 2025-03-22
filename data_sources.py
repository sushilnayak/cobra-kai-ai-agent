import json

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text


class BaseDataSource:
    """Base class for all data sources."""

    def __init__(self, source_path):
        self.source_path = source_path
        self.metadata = None

    def register_metadata(self, metadata):
        """Register metadata for this data source"""
        self.metadata = metadata

    def get_data(self):
        """Method to be implemented by child classes."""
        raise NotImplementedError("Subclasses must implement get_data()")

    def get_metadata(self):
        """Get the metadata for this data source"""
        return self.metadata

    def execute_query(self, query):
        """Execute a query against this data source"""
        raise NotImplementedError("Subclasses must implement execute_query()")


class ExcelDataSource(BaseDataSource):
    """Data source for Excel files."""

    def __init__(self, file_path, sheet_name=None):
        super().__init__(file_path)
        self.sheet_name = sheet_name

    def get_data(self):
        """
        Read data from Excel file.
        
        Returns:
            pandas.DataFrame: Data from the Excel file
        """
        try:
            if self.sheet_name:
                df = pd.read_excel(self.source_path, sheet_name=self.sheet_name)
            else:
                df = pd.read_excel(self.source_path)

            # Apply mappings if available in metadata
            if self.metadata and 'columns' in self.metadata:
                for col_meta in self.metadata['columns']:
                    if 'mapping' in col_meta and col_meta['name'] in df.columns:
                        col_name = col_meta['name']
                        # Apply mapping to convert codes to readable values
                        df[col_name] = df[col_name].map(
                            lambda x: col_meta['mapping'].get(x, x)
                        )

            return df
        except FileNotFoundError:
            print(f"Warning: Excel file not found at {self.source_path}")
            # Return empty DataFrame with sample structure for testing
            return self._get_sample_student_data()

    def get_column_info(self):
        """Return readable column information based on metadata"""
        if not self.metadata:
            return {}

        return {
            col['name']: {
                'display_name': col.get('display_name', col['name']),
                'description': col.get('description', ''),
                'type': col.get('type', 'string')
            }
            for col in self.metadata.get('columns', [])
        }

    def _get_sample_student_data(self):
        """Return sample student data for testing."""
        return pd.DataFrame({
            'std_id': [1, 2, 3, 4, 5],
            'f_nm': ['Daniel', 'Johnny', 'Miguel', 'Samantha', 'Robby'],
            'l_nm': ['LaRusso', 'Lawrence', 'Diaz', 'LaRusso', 'Keene'],
            'age': [16, 17, 16, 16, 17],
            'rnk_cd': ['BK', 'BK', 'BR', 'BR', 'BR'],
            'dojo_loc': ['Encino', 'Reseda', 'Reseda', 'Encino', 'Reseda'],
            'enrl_dt': ['2020-01-15', '2020-01-10', '2020-02-20', '2020-02-15', '2020-03-05']
        })

    def query_data(self, query_string, llm=None):
        """Query the Excel data using natural language"""
        # First get the data
        df = self.get_data()
        if df.empty:
            return df

        if not llm:
            print("Warning: LLM not provided for query_data, returning full dataset")
            return df

        # Create a prompt with metadata and query
        prompt = f"""
        Given the following Excel data structure:
        {json.dumps(self.metadata, indent=2)}
        
        The data has {len(df)} rows and columns: {", ".join(df.columns.tolist())}
        Sample data (first 3 rows):
        {df.head(3).to_json(orient='records')}
        
        Please convert this natural language query to a pandas DataFrame operation:
        "{query_string}"
        
        Return ONLY Python code using pandas that would execute this query.
        """

        # Get the pandas code from AI
        response = llm.invoke(prompt)
        pandas_code = response.content.strip()

        # Execute the code with safety measures
        try:
            # Add safety restrictions to prevent dangerous operations
            safe_globals = {
                'df': df,
                'pd': pd,
                'np': np
            }
            # Remove any imports or file operations
            if ('import' in pandas_code or
                    'open(' in pandas_code or
                    'exec(' in pandas_code or
                    'eval(' in pandas_code):
                raise ValueError("Unsafe code detected")

            # Execute the pandas operation
            result = eval(pandas_code, safe_globals)
            return result
        except Exception as e:
            print(f"Error executing pandas query: {e}")
            return pd.DataFrame({'error': [str(e)]})


class CSVDataSource(BaseDataSource):
    """Data source for CSV files."""

    def __init__(self, file_path, delimiter=',', encoding='utf-8'):
        super().__init__(file_path)
        self.delimiter = delimiter
        self.encoding = encoding

    def get_data(self):
        """
        Read data from CSV file.
        
        Returns:
            pandas.DataFrame: Data from the CSV file
        """
        try:
            df = pd.read_csv(self.source_path, delimiter=self.delimiter, encoding=self.encoding)

            # Apply mappings if available in metadata
            if self.metadata and 'columns' in self.metadata:
                for col_meta in self.metadata['columns']:
                    if 'mapping' in col_meta and col_meta['name'] in df.columns:
                        col_name = col_meta['name']
                        df[col_name] = df[col_name].map(
                            lambda x: col_meta['mapping'].get(str(x), x)
                        )

            return df
        except FileNotFoundError:
            print(f"Warning: CSV file not found at {self.source_path}")
            # Return empty DataFrame with sample structure for testing
            return self._get_sample_class_data()

    def get_column_info(self):
        """Return readable column information based on metadata"""
        if not self.metadata:
            return {}

        return {
            col['name']: {
                'display_name': col.get('display_name', col['name']),
                'description': col.get('description', ''),
                'type': col.get('type', 'string')
            }
            for col in self.metadata.get('columns', [])
        }

    def _get_sample_class_data(self):
        """Return sample class data for testing."""
        return pd.DataFrame({
            'cls_id': [1, 2, 3, 4, 5],
            'cls_nm': ['Beginner Karate', 'Advanced Karate', 'Self Defense', 'Competition Prep',
                       'Women\'s Self Defense'],
            'day_cd': ['MON', 'TUE', 'WED', 'THU', 'FRI'],
            'st_time': ['16:00', '18:00', '17:00', '19:00', '16:00'],
            'end_time': ['17:00', '19:30', '18:00', '20:30', '17:00'],
            'ins_id': [1, 2, 1, 2, 3],
            'loc_cd': ['RES', 'ENC', 'RES', 'ENC', 'NHO']
        })

    def query_data(self, query_string, llm=None):
        """Query the CSV data using natural language"""
        # First get the data
        df = self.get_data()
        if df.empty:
            return df

        if not llm:
            print("Warning: LLM not provided for query_data, returning full dataset")
            return df

        # Create a prompt with metadata and query
        prompt = f"""
        Given the following CSV data structure:
        {json.dumps(self.metadata, indent=2)}
        
        The data has {len(df)} rows and columns: {", ".join(df.columns.tolist())}
        Sample data (first 3 rows):
        {df.head(3).to_json(orient='records')}
        
        Please convert this natural language query to a pandas DataFrame operation:
        "{query_string}"
        
        Return ONLY Python code using pandas that would execute this query.
        """

        # Get the pandas code from AI
        response = llm.invoke(prompt)
        pandas_code = response.content.strip()

        # Execute the code with safety measures
        try:
            # Add safety restrictions to prevent dangerous operations
            safe_globals = {
                'df': df,
                'pd': pd,
                'np': np
            }
            # Remove any imports or file operations
            if ('import' in pandas_code or
                    'open(' in pandas_code or
                    'exec(' in pandas_code or
                    'eval(' in pandas_code):
                raise ValueError("Unsafe code detected")

            # Execute the pandas operation
            result = eval(pandas_code, safe_globals)
            return result
        except Exception as e:
            print(f"Error executing pandas query: {e}")
            return pd.DataFrame({'error': [str(e)]})


class JSONDataSource(BaseDataSource):
    """Data source for JSON files."""

    def __init__(self, file_path=None, json_url=None):
        super().__init__(file_path or json_url)
        self.file_path = file_path
        self.json_url = json_url
        if not file_path and not json_url:
            raise ValueError("Either file_path or json_url must be provided")

    def get_data(self):
        """
        Read data from JSON file or URL.
        
        Returns:
            dict or list: Data from the JSON file
        """
        try:
            if self.file_path:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                import requests
                response = requests.get(self.json_url)
                response.raise_for_status()
                data = response.json()

            # Apply field mappings if available
            if self.metadata and self.metadata.get('structure_type') == 'array':
                self._apply_mappings(data)

            return data
        except FileNotFoundError:
            print(f"Warning: JSON file not found at {self.file_path}")
            # Return sample data for testing
            return self._get_sample_instructor_data()

    def _apply_mappings(self, data):
        """Apply mappings to JSON data"""
        if not self.metadata or 'fields' not in self.metadata:
            return

        # Handle array of objects
        if isinstance(data, list):
            for item in data:
                for field_meta in self.metadata['fields']:
                    if 'mapping' in field_meta:
                        # Handle nested paths
                        path_parts = field_meta['path'].split('.')
                        target = item

                        # Navigate to the nested field
                        for i, part in enumerate(path_parts):
                            if i == len(path_parts) - 1:
                                # Apply mapping on the final field
                                if part in target and target[part] in field_meta['mapping']:
                                    target[part] = field_meta['mapping'][target[part]]
                            else:
                                # Navigate deeper if the path exists
                                if part in target and isinstance(target[part], dict):
                                    target = target[part]
                                else:
                                    break

    def _get_sample_instructor_data(self):
        """Return sample instructor data for testing."""
        return [
            {
                "ins_id": 1,
                "name": {
                    "first": "Johnny",
                    "last": "Lawrence"
                },
                "rank": "5th Degree Black Belt",
                "specialty": "KU",
                "years_experience": 35,
                "bio": "Founder of Eagle Fang Karate, former Cobra Kai student."
            },
            {
                "ins_id": 2,
                "name": {
                    "first": "Daniel",
                    "last": "LaRusso"
                },
                "rank": "5th Degree Black Belt",
                "specialty": "KT",
                "years_experience": 35,
                "bio": "Trained by Mr. Miyagi, All Valley Champion 1984."
            },
            {
                "ins_id": 3,
                "name": {
                    "first": "Aisha",
                    "last": "Robinson"
                },
                "rank": "2nd Degree Black Belt",
                "specialty": "SD",
                "years_experience": 3,
                "bio": "Top female competitor, specializes in teaching women's classes."
            }
        ]

    def query_data(self, query_string, llm=None):
        """Extract relevant data from JSON based on natural language query"""
        # First get the data
        data = self.get_data()
        if not data:
            return {}

        if not llm:
            print("Warning: LLM not provided for query_data, returning full dataset")
            return data

        # Create a prompt with metadata and query
        prompt = f"""
        Given the following JSON data structure:
        {json.dumps(self.metadata, indent=2)}
        
        And the data (showing a sample):
        {json.dumps(data[:2] if isinstance(data, list) else data, indent=2)}
        
        Please extract information from this data to answer the following query:
        "{query_string}"
        
        Return a JSON object with the following structure:
        {{
            "result": [] // Array containing the answer to the query
            "count": 0,  // Number of results
            "fields": [] // List of fields in the result
        }}
        """

        # Get the extraction logic from AI
        response = llm.invoke(prompt)
        try:
            result = json.loads(response.content.strip())
            return result
        except Exception as e:
            print(f"Error processing JSON query: {e}")
            return {"error": str(e), "data": data[:1] if isinstance(data, list) else data}


class SQLDataSource(BaseDataSource):
    """Data source for SQL databases."""

    def __init__(self, connection_string, dialect="sqlite"):
        super().__init__(connection_string)
        self.connection_string = connection_string
        self.dialect = dialect

    def get_data(self, table_name=None, limit=100):
        """
        Get data from specified table or tables.
        
        Args:
            table_name: Optional table name to query
            limit: Max number of rows to return
            
        Returns:
            pandas.DataFrame: Data from the table
        """
        try:
            engine = create_engine(self.connection_string)
            if table_name:
                query = f"SELECT * FROM {table_name} LIMIT {limit}"
                return pd.read_sql(query, engine)
            else:
                # If no table specified, return metadata
                return self.metadata
        except Exception as e:
            print(f"Warning: SQL query failed: {e}")
            # Return sample data for testing
            return self._get_sample_financial_data()

    def execute_query(self, query):
        """Execute a SQL query"""
        try:
            engine = create_engine(self.connection_string)
            with engine.connect() as conn:
                result = conn.execute(text(query))
                columns = result.keys()
                rows = [dict(zip(columns, row)) for row in result.fetchall()]
                return {"columns": columns, "rows": rows}
        except Exception as e:
            print(f"Error executing SQL query: {e}")
            return {"error": str(e)}

    def _get_sample_financial_data(self):
        """Return sample financial data for testing."""
        return pd.DataFrame({
            'month': ['January', 'February', 'March', 'April', 'May'],
            'year': [2023, 2023, 2023, 2023, 2023],
            'loc_cd': ['RES', 'ENC', 'NHO', 'RES', 'ENC'],
            'revenue': [45000, 52000, 38000, 47500, 54000],
            'expenses': [32000, 35000, 28000, 33500, 36000],
            'new_students': [15, 18, 12, 16, 20],
            'tot_students': [125, 143, 105, 138, 160]
        })
