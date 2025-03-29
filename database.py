import sqlite3
import json
import os
from datetime import datetime

class ConversationDB:
    def __init__(self, db_path="conversations.db"):
        """Initialize the database connection and create tables if they don't exist."""
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self.connect()
        self.create_tables()
    
    def connect(self):
        """Connect to the SQLite database."""
        # Create the database directory if it doesn't exist
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
            
        self.conn = sqlite3.connect(self.db_path)
        # Enable foreign keys
        self.conn.execute("PRAGMA foreign_keys = ON")
        # Return rows as dictionaries
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
    
    def create_tables(self):
        """Create the necessary tables if they don't exist."""
        # Conversations table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            prompt TEXT NOT NULL,
            response TEXT,
            metadata TEXT
        )
        ''')
        
        # Token usage table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS token_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL,
            input_tokens INTEGER,
            output_tokens INTEGER,
            total_tokens INTEGER,
            reasoning_tokens INTEGER,
            accepted_prediction_tokens INTEGER,
            rejected_prediction_tokens INTEGER,
            execution_time REAL,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
        )
        ''')
        
        self.conn.commit()
    
    def save_conversation(self, model_name, prompt, response, metadata=None):
        """
        Save a conversation to the database.
        
        Args:
            model_name (str): The name of the model used
            prompt (str): The user's prompt
            response (str): The model's response
            metadata (dict): Additional metadata about the conversation
        
        Returns:
            int: The ID of the inserted conversation
        """
        metadata_json = json.dumps(metadata) if metadata else None
        
        self.cursor.execute(
            "INSERT INTO conversations (model_name, prompt, response, metadata, timestamp) VALUES (?, ?, ?, ?, ?)",
            (model_name, prompt, response, metadata_json, datetime.now().isoformat())
        )
        
        conversation_id = self.cursor.lastrowid
        self.conn.commit()
        return conversation_id
    
    def save_token_usage(self, conversation_id, input_tokens=0, output_tokens=0, total_tokens=None, 
                         reasoning_tokens=None, accepted_prediction_tokens=None, 
                         rejected_prediction_tokens=None, execution_time=None):
        """
        Save token usage information for a conversation.
        
        Args:
            conversation_id (int): The ID of the conversation
            input_tokens (int): Number of input tokens
            output_tokens (int): Number of output tokens
            total_tokens (int): Total number of tokens (if provided separately)
            reasoning_tokens (int): Number of reasoning tokens (OpenAI models)
            accepted_prediction_tokens (int): Number of accepted prediction tokens (OpenAI models)
            rejected_prediction_tokens (int): Number of rejected prediction tokens (OpenAI models)
            execution_time (float): Total execution time in seconds
        """
        # Calculate total if not provided
        if total_tokens is None:
            total_tokens = input_tokens + output_tokens
            
        self.cursor.execute(
            """
            INSERT INTO token_usage (
                conversation_id, input_tokens, output_tokens, total_tokens,
                reasoning_tokens, accepted_prediction_tokens, rejected_prediction_tokens,
                execution_time
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (conversation_id, input_tokens, output_tokens, total_tokens,
             reasoning_tokens, accepted_prediction_tokens, rejected_prediction_tokens,
             execution_time)
        )
        
        self.conn.commit()
    
    def get_conversation(self, conversation_id):
        """
        Retrieve a conversation by ID.
        
        Args:
            conversation_id (int): The ID of the conversation to retrieve
            
        Returns:
            dict: The conversation data or None if not found
        """
        self.cursor.execute(
            """
            SELECT c.*, t.input_tokens, t.output_tokens, t.total_tokens
            FROM conversations c
            LEFT JOIN token_usage t ON c.id = t.conversation_id
            WHERE c.id = ?
            """, 
            (conversation_id,)
        )
        
        row = self.cursor.fetchone()
        if row:
            result = dict(row)
            if result['metadata']:
                result['metadata'] = json.loads(result['metadata'])
                
            # Add reasoning tokens info if available
            if 'reasoning_tokens' in result and result['reasoning_tokens'] is not None:
                result['reasoning_tokens_info'] = {
                    'reasoning_tokens': result['reasoning_tokens'],
                    'accepted_prediction_tokens': result['accepted_prediction_tokens'],
                    'rejected_prediction_tokens': result['rejected_prediction_tokens']
                }
                
            return result
        return None
    
    def get_all_conversations(self, limit=100, offset=0):
        """
        Retrieve all conversations with pagination.
        
        Args:
            limit (int): Maximum number of conversations to retrieve
            offset (int): Number of conversations to skip
            
        Returns:
            list: List of conversation dictionaries
        """
        self.cursor.execute(
            """
            SELECT c.*, t.input_tokens, t.output_tokens, t.total_tokens, 
                   t.reasoning_tokens, t.accepted_prediction_tokens, t.rejected_prediction_tokens,
                   t.execution_time
            FROM conversations c
            LEFT JOIN token_usage t ON c.id = t.conversation_id
            ORDER BY c.timestamp DESC
            LIMIT ? OFFSET ?
            """, 
            (limit, offset)
        )
        
        rows = self.cursor.fetchall()
        result = []
        for row in rows:
            conversation = dict(row)
            if conversation['metadata']:
                conversation['metadata'] = json.loads(conversation['metadata'])
                
            # Add reasoning tokens info if available
            if 'reasoning_tokens' in conversation and conversation['reasoning_tokens'] is not None:
                conversation['reasoning_tokens_info'] = {
                    'reasoning_tokens': conversation['reasoning_tokens'],
                    'accepted_prediction_tokens': conversation['accepted_prediction_tokens'],
                    'rejected_prediction_tokens': conversation['rejected_prediction_tokens']
                }
                
            result.append(conversation)
        
        return result
    
    def search_conversations(self, query, limit=100, offset=0):
        """
        Search conversations by prompt or response content.
        
        Args:
            query (str): Search query
            limit (int): Maximum number of conversations to retrieve
            offset (int): Number of conversations to skip
            
        Returns:
            list: List of matching conversation dictionaries
        """
        search_param = f"%{query}%"
        self.cursor.execute(
            """
            SELECT c.*, t.input_tokens, t.output_tokens, t.total_tokens,
                   t.reasoning_tokens, t.accepted_prediction_tokens, t.rejected_prediction_tokens,
                   t.execution_time
            FROM conversations c
            LEFT JOIN token_usage t ON c.id = t.conversation_id
            WHERE c.prompt LIKE ? OR c.response LIKE ?
            ORDER BY c.timestamp DESC
            LIMIT ? OFFSET ?
            """, 
            (search_param, search_param, limit, offset)
        )
        
        rows = self.cursor.fetchall()
        result = []
        for row in rows:
            conversation = dict(row)
            if conversation['metadata']:
                conversation['metadata'] = json.loads(conversation['metadata'])
                
            # Add reasoning tokens info if available
            if 'reasoning_tokens' in conversation and conversation['reasoning_tokens'] is not None:
                conversation['reasoning_tokens_info'] = {
                    'reasoning_tokens': conversation['reasoning_tokens'],
                    'accepted_prediction_tokens': conversation['accepted_prediction_tokens'],
                    'rejected_prediction_tokens': conversation['rejected_prediction_tokens']
                }
                
            result.append(conversation)
        
        return result
    
    def get_stats(self):
        """
        Get usage statistics from the database.
        
        Returns:
            dict: Statistics about conversations and token usage
        """
        stats = {}
        
        # Total conversations
        self.cursor.execute("SELECT COUNT(*) as count FROM conversations")
        stats['total_conversations'] = self.cursor.fetchone()['count']
        
        # Total tokens by model
        self.cursor.execute(
            """
            SELECT c.model_name, 
                   SUM(t.input_tokens) as total_input, 
                   SUM(t.output_tokens) as total_output,
                   SUM(t.total_tokens) as total
            FROM conversations c
            JOIN token_usage t ON c.id = t.conversation_id
            GROUP BY c.model_name
            """
        )
        stats['tokens_by_model'] = [dict(row) for row in self.cursor.fetchall()]
        
        # Conversations by date
        self.cursor.execute(
            """
            SELECT DATE(timestamp) as date, COUNT(*) as count
            FROM conversations
            GROUP BY DATE(timestamp)
            ORDER BY date DESC
            LIMIT 30
            """
        )
        stats['conversations_by_date'] = [dict(row) for row in self.cursor.fetchall()]
        
        return stats
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None
