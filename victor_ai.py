import json
import os

import pandas as pd
from google.cloud import aiplatform
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_google_vertexai import ChatVertexAI

from data_sources import (
    ExcelDataSource,
    CSVDataSource,
    JSONDataSource,
    SQLDataSource
)


class VictorAI:
    def __init__(self):
        """Initialize the Victor AI assistant for Karate School Franchise."""

        # Initialize Vertex AI
        try:
            project_id = os.getenv("GCP_PROJECT", "your-project-id")
            print(f"Initializing Vertex AI with project ID: {project_id}")
            aiplatform.init(project=project_id)
            self.llm = ChatVertexAI(
                model_name="gemini-1.0-pro",
                temperature=0.2,
                max_output_tokens=1024,
                top_p=0.8,
                top_k=40
            )
            print("Vertex AI initialized successfully")
        except Exception as e:
            print(f"ERROR initializing Vertex AI: {e}")
            self.llm = None

        # Initialize data sources with metadata
        self.data_sources = {}
        self._initialize_data_sources()

        # Track available tables for database sources
        self.db_tables = {
            # Example of how to define tables for a database source
            # "database_source_name": {
            #     "students": "Information about student enrollment, demographics, and progress",
            #     "classes": "Class schedules, instructors, and capacity",
            #     "payments": "Financial transactions and revenue"
            # }
        }

        # Create prompt templates
        self.response_prompt = PromptTemplate(
            input_variables=["query", "context"],
            template="""
            You are Victor AI, an intelligent assistant for the Kobra Kai Karate School Franchise.
            
            Your knowledge comes from various data sources including student records, class schedules,
            instructor information, and financial data.
            
            USER QUERY: {query}
            
            RELEVANT CONTEXT: {context}
            
            Please provide a helpful, informative, and friendly response. Use a chain-of-thought
            approach to explain your reasoning step by step. If information is not available in the
            context, indicate that politely without making up details.
            
            IMPORTANT FORMATTING GUIDELINES:
            1. Use clear section headers (ending with a colon) for different parts of your response
            2. Use **bold** for key terms and important information
            3. Break your response into well-structured paragraphs
            4. Keep your sentences clear and concise
            5. If providing steps or a list, number them clearly
            """
        )

        self.follow_up_prompt = PromptTemplate(
            input_variables=["query", "response"],
            template="""
            Based on the following conversation, suggest 3 relevant follow-up questions that the user
            might want to ask next. Make them specific to the Kobra Kai Karate School Franchise context.
            
            USER QUERY: {query}
            
            YOUR RESPONSE: {response}
            
            Three potential follow-up questions:
            """
        )

        # Create runnable chains
        if self.llm:
            # Using runnable sequence with RunnablePassthrough
            self.response_chain = (
                    {"query": RunnablePassthrough(), "context": lambda x: self._get_relevant_context(x)}
                    | self.response_prompt
                    | self.llm
            )

            self.follow_up_chain = self.follow_up_prompt | self.llm

    def _initialize_data_sources(self):
        """Initialize data sources with metadata from config files."""
        # Define the data sources with their corresponding metadata files
        sources_config = {
            "students": {
                "source_class": ExcelDataSource,
                "metadata_file": "config/student_metadata.json"
            },
            "classes": {
                "source_class": CSVDataSource,
                "metadata_file": "config/class_metadata.json"
            },
            "instructors": {
                "source_class": JSONDataSource,
                "metadata_file": "config/instructor_metadata.json"
            },
            "finances": {
                "source_class": CSVDataSource,
                "metadata_file": "config/finances_metadata.json"
            }
        }

        # Initialize each data source with its metadata
        for source_name, config in sources_config.items():
            try:
                # Load metadata
                if os.path.exists(config["metadata_file"]):
                    with open(config["metadata_file"], 'r') as f:
                        metadata = json.load(f)

                    # Create data source instance
                    source_instance = config["source_class"](metadata["file_path"])

                    # Register metadata with the data source
                    source_instance.register_metadata(metadata)

                    # Add to data sources dictionary
                    self.data_sources[source_name] = source_instance
                    print(f"Initialized {source_name} data source with metadata")
                else:
                    # Fallback to default initialization if metadata file doesn't exist
                    if source_name == "students":
                        file_path = "data/students.xlsx"
                    elif source_name == "classes":
                        file_path = "data/classes.csv"
                    elif source_name == "instructors":
                        file_path = "data/instructors.json"
                    elif source_name == "finances":
                        file_path = "data/finances.csv"
                    else:
                        continue

                    source_instance = config["source_class"](file_path)
                    self.data_sources[source_name] = source_instance
                    print(
                        f"Initialized {source_name} data source without metadata (file not found: {config['metadata_file']})")
            except Exception as e:
                print(f"Error initializing {source_name} data source: {e}")
                # Create with default path as fallback
                if source_name == "students":
                    self.data_sources[source_name] = ExcelDataSource("data/students.xlsx")
                elif source_name == "classes":
                    self.data_sources[source_name] = CSVDataSource("data/classes.csv")
                elif source_name == "instructors":
                    self.data_sources[source_name] = JSONDataSource("data/instructors.json")
                elif source_name == "finances":
                    self.data_sources[source_name] = CSVDataSource("data/finances.csv")

    def _get_relevant_context(self, query):
        """
        Retrieve relevant information from data sources based on the query.
        Uses metadata to provide better context and more accurate results.
        """
        context = []
        query_lower = query.lower()

        # Track if we found any relevant data
        found_data = False

        # Extract metadata for context if available
        metadata_overview = []
        for source_name, source in self.data_sources.items():
            if hasattr(source, 'metadata') and source.metadata:
                if 'description' in source.metadata:
                    metadata_overview.append(f"- {source_name.capitalize()}: {source.metadata['description']}")

        if metadata_overview:
            context.append("Available data sources:")
            context.extend(metadata_overview)

        # Students data
        if any(word in query_lower for word in ["student", "students", "enrollment", "enrolled", "demographics"]):
            try:
                student_data = self.data_sources["students"].get_data()
                if not student_data.empty:
                    # Add column descriptions if available
                    if hasattr(self.data_sources["students"], 'metadata') and self.data_sources["students"].metadata:
                        col_info = {}
                        if 'columns' in self.data_sources["students"].metadata:
                            for col in self.data_sources["students"].metadata['columns']:
                                display_name = col.get('display_name', col['name'])
                                col_info[col['name']] = display_name

                            # Provide context with better column names
                            student_dict = student_data.head(5).to_dict()
                            for col_name, values in student_dict.items():
                                if col_name in col_info:
                                    student_dict[col_info[col_name]] = student_dict.pop(col_name)

                            context.append(f"Student data (showing 5 of {len(student_data)} records):")
                            context.append(f"{student_dict}")
                    else:
                        context.append(f"Student data: {student_data.head(5).to_dict()}")

                    context.append(f"Total students enrolled: {len(student_data)}")
                    found_data = True
                else:
                    context.append("Student data: No student records found.")
            except Exception as e:
                context.append(f"Error retrieving student data: {e}")

        # Classes data
        if any(word in query_lower for word in ["class", "classes", "schedule", "schedules", "program"]):
            try:
                class_data = self.data_sources["classes"].get_data()
                if not class_data.empty:
                    # Add column descriptions if available
                    if hasattr(self.data_sources["classes"], 'metadata') and self.data_sources["classes"].metadata:
                        col_info = {}
                        if 'columns' in self.data_sources["classes"].metadata:
                            for col in self.data_sources["classes"].metadata['columns']:
                                display_name = col.get('display_name', col['name'])
                                col_info[col['name']] = display_name

                            # Provide context with better column names
                            class_dict = class_data.head(5).to_dict()
                            for col_name, values in list(class_dict.items()):
                                if col_name in col_info:
                                    class_dict[col_info[col_name]] = class_dict.pop(col_name)

                            context.append(f"Class data (showing 5 of {len(class_data)} records):")
                            context.append(f"{class_dict}")
                    else:
                        context.append(f"Class data: {class_data.head(5).to_dict()}")
                    found_data = True
                else:
                    context.append("Class data: No class records found.")
            except Exception as e:
                context.append(f"Error retrieving class data: {e}")

        # Instructors data
        if any(word in query_lower for word in ["instructor", "instructors", "sensei", "teacher", "staff"]):
            try:
                instructor_data = self.data_sources["instructors"].get_data()
                if instructor_data:
                    # Format instructor data with metadata if available
                    if hasattr(self.data_sources["instructors"], 'metadata') and self.data_sources[
                        "instructors"].metadata:
                        formatted_data = []
                        for instructor in instructor_data[:3]:
                            formatted_instructor = {}
                            if 'fields' in self.data_sources["instructors"].metadata:
                                for field in self.data_sources["instructors"].metadata['fields']:
                                    field_name = field['name']
                                    display_name = field.get('display_name', field_name)
                                    # Handle nested fields like name.first
                                    if '.' in field_name:
                                        parts = field_name.split('.')
                                        value = instructor
                                        for part in parts:
                                            value = value.get(part, {}) if isinstance(value, dict) else None
                                        if value is not None:
                                            formatted_instructor[display_name] = value
                                    else:
                                        if field_name in instructor:
                                            formatted_instructor[display_name] = instructor[field_name]
                            formatted_data.append(formatted_instructor or instructor)

                        context.append(f"Instructor data (showing 3 of {len(instructor_data)} records):")
                        context.append(f"{json.dumps(formatted_data)}")
                    else:
                        context.append(f"Instructor data: {json.dumps(instructor_data[:3])}")
                    found_data = True
                else:
                    context.append("Instructor data: No instructor records found.")
            except Exception as e:
                context.append(f"Error retrieving instructor data: {e}")

        # Financial data
        if any(word in query_lower for word in
               ["revenue", "financial", "finances", "money", "income", "sales", "performance"]):
            try:
                financial_data = self.data_sources["finances"].get_data()
                if not financial_data.empty:
                    # Add column descriptions if available
                    if hasattr(self.data_sources["finances"], 'metadata') and self.data_sources["finances"].metadata:
                        col_info = {}
                        if 'columns' in self.data_sources["finances"].metadata:
                            for col in self.data_sources["finances"].metadata['columns']:
                                display_name = col.get('display_name', col['name'])
                                col_info[col['name']] = display_name

                            # Provide context with better column names
                            finance_dict = financial_data.head(5).to_dict()
                            for col_name, values in list(finance_dict.items()):
                                if col_name in col_info:
                                    finance_dict[col_info[col_name]] = finance_dict.pop(col_name)

                            context.append(f"Financial data (showing 5 of {len(financial_data)} records):")
                            context.append(f"{finance_dict}")
                    else:
                        context.append(f"Financial data: {financial_data.head(5).to_dict()}")
                    found_data = True
                else:
                    context.append("Financial data: No financial records found.")
            except Exception as e:
                context.append(f"Error retrieving financial data: {e}")

        # If no specific context was found
        if not found_data:
            context.append("No specific data was found for your query. I can help with information about:")
            context.append("- Student enrollment and demographics")
            context.append("- Class schedules and programs")
            context.append("- Instructor details and specialties")
            context.append("- Financial performance and metrics")

        return "\n".join(context)

    def generate_response(self, query):
        """Generate a response to the user query using the chain of thought approach."""
        if not self.llm:
            return "I'm currently experiencing connection issues with my AI backend. Please check the application settings and ensure Google Cloud credentials are configured correctly."

        try:
            # Generate response using the runnable chain
            response = self.response_chain.invoke(query)

            # Handle different response formats from newer LangChain versions
            if hasattr(response, 'content'):
                return response.content
            return response
        except Exception as e:
            # Return a clear error message instead of a dummy response
            return f"I encountered an error while processing your request: {str(e)}\n\nPlease try again or contact support if the issue persists."

    def suggest_follow_ups(self, query, response):
        """Generate suggested follow-up questions based on the conversation."""
        if not self.llm:
            # Return user-friendly follow-up suggestions instead of technical ones
            return [
                "How can I get started with Kobra Kai Karate?",
                "What programs do you offer for beginners?",
                "Tell me about your instructors"
            ]

        try:
            # Generate follow-up suggestions using the runnable chain
            follow_up_result = self.follow_up_chain.invoke({"query": query, "response": response})

            # Handle different response formats
            if hasattr(follow_up_result, 'content'):
                follow_up_text = follow_up_result.content
            else:
                follow_up_text = follow_up_result

            # Parse the result into a list of questions
            follow_ups = []
            for line in follow_up_text.strip().split("\n"):
                # Clean up the numbered format (e.g., "1. Question" -> "Question")
                clean_line = line.strip()
                if clean_line and any(clean_line.startswith(str(i)) for i in range(1, 10)):
                    question = clean_line.split(".", 1)[1].strip() if "." in clean_line else clean_line
                    follow_ups.append(question)

            # Limit to 3 follow-ups
            return follow_ups[:3]
        except Exception as e:
            # Return error-related follow-ups
            return [
                "How do I fix the Vertex AI connection?",
                "What credentials do I need for Google Cloud?",
                "Can you explain how to set up the environment variables?"
            ]

    def add_database_source(self, name, connection_string, tables_info):
        """
        Add a database source with information about its tables.
        
        Args:
            name: Name to reference this database source
            connection_string: SQLAlchemy connection string
            tables_info: Dict mapping table names to descriptions
        """
        self.data_sources[name] = SQLDataSource(connection_string)
        self.db_tables[name] = tables_info

    def _get_relevant_tables(self, query):
        """
        Determine which database tables are relevant to a query.
        This is a simple keyword-based approach that could be enhanced with embeddings.
        
        Returns a dict mapping source_name -> list of relevant table names
        """
        relevant_tables = {}
        query_lower = query.lower()

        for source_name, tables in self.db_tables.items():
            matching_tables = []

            for table_name, description in tables.items():
                # Check if table name is mentioned directly
                if table_name.lower() in query_lower:
                    matching_tables.append(table_name)
                    continue

                # Check if terms in the description match
                description_words = set(description.lower().split())
                if any(word in query_lower for word in description_words):
                    matching_tables.append(table_name)

            if matching_tables:
                relevant_tables[source_name] = matching_tables

        return relevant_tables
