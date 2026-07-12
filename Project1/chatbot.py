import utils
import responses

def display_welcome_screen() -> None:
    """
    Displays the chatbot welcome banner and starter instructions.
    """
    print("===================================")
    print("      Rule-Based AI Chatbot")
    print("===================================")
    print("Hello! I am your AI assistant.")
    print("Type 'help' to see available commands.")
    print("Type 'bye', 'exit', or 'quit' to quit.")
    print("===================================\n")

def display_history(chat_history: list) -> None:
    """
    Prints a formatted transcript of the conversation history.
    
    Args:
        chat_history (list): A list of tuples containing (turn_number, user_message, bot_response).
    """
    if not chat_history:
        print("Bot: Your conversation history is currently empty.")
        print()
        return

    print("\n=============================================")
    print("                CHAT HISTORY")
    print("=============================================")
    for turn, user_msg, bot_msg in chat_history:
        print(f"Turn #{turn}")
        print(f"  User: {user_msg}")
        print(f"  Bot:  {bot_msg}")
        print("-" * 45)
    print("=============================================\n")

def main() -> None:
    """
    Main loop governing the chatbot application. Handles turns,
    invokes input parsing, manages state (turn count & history),
    and captures exceptions to prevent crashes.
    """
    # Start with a clean console
    utils.clear_screen()
    display_welcome_screen()
    
    # Session state
    chat_history = []
    turn_number = 1

    while True:
        try:
            # Display conversation number and accept raw user input
            prompt = f"User #{turn_number}: "
            user_input_raw = input(prompt)
            
            # Clean and normalize input
            cleaned_input = utils.clean_input(user_input_raw)
            
            # Handle empty input gracefully by prompting again without incrementing the counter
            if not cleaned_input:
                print("Bot: It looks like you didn't type anything. How can I help you?")
                print()
                continue
            
            # Handle 'clear' command directly (console utility)
            if cleaned_input == "clear":
                utils.clear_screen()
                display_welcome_screen()
                continue
            
            # Handle 'history' command (session history utility)
            if cleaned_input == "history":
                display_history(chat_history)
                continue
            
            # Get matching response using the rule engine
            response = responses.get_response(cleaned_input)
            
            # Print response
            print(f"Bot: {response}")
            print()
            
            # Record current turn in history using the raw input for fidelity
            chat_history.append((turn_number, user_input_raw, response))
            
            # Terminate loop if the user matched a goodbye rule
            if cleaned_input in ("bye", "exit", "quit"):
                break
                
            # Move to next conversation turn
            turn_number += 1

        except KeyboardInterrupt:
            # Gracefully handle console interrupts (Ctrl+C)
            print("\n\nBot: Program interrupted. Goodbye! Have a great day!")
            break
        except Exception as err:
            # Generic catch-all to prevent application crash
            print(f"\nBot: An unexpected error occurred: {err}")
            print("Please try entering another command.")
            print()

if __name__ == "__main__":
    main()
