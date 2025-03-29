import argparse
import sys
from database import ConversationDB
from tabulate import tabulate
import json

def format_text(text, max_length=20, truncate_from='start'):
    """Format text for display by truncating if too long.
    Truncate from 'start' (beginning) or 'end'."""
    if not text:
        return ""
    if len(text) > max_length:
        if truncate_from == 'start':
            return text[:max_length].replace("\n","") + "..."
        elif truncate_from == 'end':
            return "..." + text[-max_length:].replace("\n","")
    return text

def view_all_conversations(db, limit=20, offset=0, detailed=False):
    """Display all conversations with pagination"""
    conversations = db.get_all_conversations(limit=limit, offset=offset)
    
    if not conversations:
        print("No conversations found.")
        return
    
    if detailed:
        for i, conv in enumerate(conversations):
            print(f"\n=== Conversation {offset + i + 1} (ID: {conv['id']}) ===")
            print(f"Model: {conv['model_name']}")
            print(f"Time: {conv['timestamp']}")
            print(f"Input tokens: {conv.get('input_tokens', 'N/A')}")
            print(f"Output tokens: {conv.get('output_tokens', 'N/A')}")
            print(f"Total tokens: {conv.get('total_tokens', 'N/A')}")
            print(f"Execution time: {conv.get('execution_time', 'N/A')} seconds")
            
            # Display reasoning tokens info if available
            if 'reasoning_tokens_info' in conv:
                print("\nReasoning tokens info:")
                info = conv['reasoning_tokens_info']
                print(f"  Reasoning tokens: {info.get('reasoning_tokens', 'N/A')}")
                print(f"  Accepted prediction tokens: {info.get('accepted_prediction_tokens', 'N/A')}")
                print(f"  Rejected prediction tokens: {info.get('rejected_prediction_tokens', 'N/A')}")
            
            print("\nPrompt:")
            print(conv['prompt'])
            print("\nResponse:")
            print(conv['response'])
            
            if conv['metadata']:
                print("\nMetadata:")
                for key, value in conv['metadata'].items():
                    print(f"  {key}: {value}")
            
            print("\n" + "=" * 80)
    else:
        # Create a simplified table view
        table_data = []
        for conv in conversations:
            table_data.append([
                conv['id'],
                conv['model_name'],
                conv['timestamp'],
                format_text(conv['prompt'], truncate_from='start'),
                format_text(conv['response'], truncate_from='end'),
                f"{conv.get('input_tokens', 'N/A')} / {conv.get('output_tokens', 'N/A')}",
                f"{conv.get('execution_time', 'N/A'):.2f}s" if conv.get('execution_time') else 'N/A'
            ])
        
        headers = ["ID", "Model", "Timestamp", "Prompt", "Response", "Tokens (In/Out)", "Time"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        print(f"\nShowing conversations {offset+1}-{offset+len(conversations)}. Use --offset to see more.")

def view_conversation(db, conversation_id):
    """Display a single conversation in detail"""
    conv = db.get_conversation(conversation_id)
    
    if not conv:
        print(f"Conversation with ID {conversation_id} not found.")
        return
    
    print(f"\n=== Conversation {conv['id']} ===")
    print(f"Model: {conv['model_name']}")
    print(f"Time: {conv['timestamp']}")
    print(f"Input tokens: {conv.get('input_tokens', 'N/A')}")
    print(f"Output tokens: {conv.get('output_tokens', 'N/A')}")
    print(f"Total tokens: {conv.get('total_tokens', 'N/A')}")
    print(f"Execution time: {conv.get('execution_time', 'N/A')} seconds")
    
    # Display reasoning tokens info if available
    if 'reasoning_tokens_info' in conv:
        print("\nReasoning tokens info:")
        info = conv['reasoning_tokens_info']
        print(f"  Reasoning tokens: {info.get('reasoning_tokens', 'N/A')}")
        print(f"  Accepted prediction tokens: {info.get('accepted_prediction_tokens', 'N/A')}")
        print(f"  Rejected prediction tokens: {info.get('rejected_prediction_tokens', 'N/A')}")
    
    print("\nPrompt:")
    print(format_text(conv['prompt'], truncate_from='start'))
    
    print("\nResponse:")
    print(format_text(conv['response'], truncate_from='end'))
    
    if conv['metadata']:
        print("\nMetadata:")
        for key, value in conv['metadata'].items():
            print(f"  {key}: {value}")

def search_conversations(db, query, limit=20, offset=0):
    """Search for conversations containing the query"""
    conversations = db.search_conversations(query, limit=limit, offset=offset)
    
    if not conversations:
        print(f"No conversations found matching '{query}'.")
        return
    
    # Create a simplified table view
    table_data = []
    for conv in conversations:
        table_data.append([
            conv['id'],
            conv['model_name'],
            conv['timestamp'],
            format_text(conv['prompt'], truncate_from='start'),
            format_text(conv['response'], truncate_from='end'),
            f"{conv.get('input_tokens', 'N/A')} / {conv.get('output_tokens', 'N/A')}",
            f"{conv.get('execution_time', 'N/A'):.2f}s" if conv.get('execution_time') else 'N/A'
        ])
    
    headers = ["ID", "Model", "Timestamp", "Prompt", "Response", "Tokens (In/Out)", "Time"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    print(f"\nFound {len(conversations)} conversations matching '{query}'.")

def view_stats(db):
    """Display usage statistics"""
    stats = db.get_stats()
    
    print("\n=== Conversation Statistics ===")
    print(f"Total conversations: {stats['total_conversations']}")
    
    if stats['tokens_by_model']:
        print("\nToken usage by model:")
        table_data = []
        for model_stats in stats['tokens_by_model']:
            table_data.append([
                model_stats['model_name'],
                model_stats['total_input'] or 0,
                model_stats['total_output'] or 0,
                model_stats['total'] or 0
            ])
        
        headers = ["Model", "Input Tokens", "Output Tokens", "Total Tokens"]
        print(tabulate(table_data, headers=headers, tablefmt="simple"))
    
    if stats['conversations_by_date']:
        print("\nConversations by date (last 30 days):")
        table_data = []
        for date_stats in stats['conversations_by_date']:
            table_data.append([
                date_stats['date'],
                date_stats['count']
            ])
        
        headers = ["Date", "Count"]
        print(tabulate(table_data, headers=headers, tablefmt="simple"))

def main():
    parser = argparse.ArgumentParser(description="View and manage LLM conversations")
    parser.add_argument("--db", default="conversations.db", help="Path to the database file")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # List all conversations
    list_parser = subparsers.add_parser("list", help="List all conversations")
    list_parser.add_argument("--limit", type=int, default=20, help="Number of conversations to show")
    list_parser.add_argument("--offset", type=int, default=0, help="Offset for pagination")
    list_parser.add_argument("--detailed", action="store_true", help="Show detailed view")
    
    # View a specific conversation
    view_parser = subparsers.add_parser("view", help="View a specific conversation")
    view_parser.add_argument("id", type=int, help="Conversation ID to view")
    
    # Search conversations
    search_parser = subparsers.add_parser("search", help="Search conversations")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--limit", type=int, default=20, help="Number of results to show")
    search_parser.add_argument("--offset", type=int, default=0, help="Offset for pagination")
    
    # View statistics
    subparsers.add_parser("stats", help="View usage statistics")
    
    args = parser.parse_args()
    
    # Initialize database connection
    db = ConversationDB(args.db)
    
    try:
        if args.command == "list":
            view_all_conversations(db, limit=args.limit, offset=args.offset, detailed=args.detailed)
        elif args.command == "view":
            view_conversation(db, args.id)
        elif args.command == "search":
            search_conversations(db, args.query, limit=args.limit, offset=args.offset)
        elif args.command == "stats":
            view_stats(db)
        else:
            # Default to listing if no command is provided
            view_all_conversations(db)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    main()
