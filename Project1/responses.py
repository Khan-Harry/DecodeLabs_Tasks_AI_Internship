import random
import utils

# Predefined jokes
JOKES = [
    "Why do programmers wear glasses? Because they can't C#!",
    "There are 10 types of people in the world: those who understand binary, and those who don't.",
    "Why did the programmer quit his job? Because he didn't get arrays.",
    "How many programmers does it take to change a light bulb? None, that's a hardware problem.",
    "What is a programmer's favorite hangout place? Foo Bar!",
    "Why do computer scientists prefer dark mode? Because light attracts bugs!"
]

# Predefined motivational quotes
MOTIVATIONAL_QUOTES = [
    "The only way to do great work is to love what you do. - Steve Jobs",
    "Success is not final, failure is not fatal: it is the courage to continue that counts. - Winston Churchill",
    "Believe you can and you're halfway there. - Theodore Roosevelt",
    "It always seems impossible until it's done. - Nelson Mandela",
    "Don't comment bad code - rewrite it. - Brian Kernighan",
    "First, solve the problem. Then, write the code. - John Johnson"
]

def get_help_message() -> str:
    """
    Returns a formatted string containing all available chatbot commands.
    
    Returns:
        str: The help message with available commands.
    """
    help_text = (
        "=========================================\n"
        "           AVAILABLE COMMANDS\n"
        "=========================================\n"
        "  - hi / hello / hey          : Greet the chatbot\n"
        "  - how are you               : Ask how the chatbot is doing\n"
        "  - what is your name         : Ask for the chatbot's name\n"
        "  - who are you               : Ask for the chatbot's identity\n"
        "  - time / current time       : Display the current system time\n"
        "  - date / today              : Display today's date\n"
        "  - tell me a joke            : Get a fun programming joke\n"
        "  - motivate me               : Get an inspiring quote\n"
        "  - who created you           : Learn about the chatbot's creator\n"
        "  - history                   : View the full conversation log\n"
        "  - clear                     : Clear the console screen\n"
        "  - help                      : Show this command list\n"
        "  - bye / exit / quit         : Exit the chatbot program\n"
        "========================================="
    )
    return help_text

def get_response(cleaned_input: str) -> str:
    """
    Core rule-based response engine. Uses standard Python if-elif-else
    control flow to match user inputs to appropriate pre-defined responses.
    
    Args:
        cleaned_input (str): The normalized, lowercase user input string.
        
    Returns:
        str: The chatbot's response text.
    """
    # 1. Greetings
    if cleaned_input in ("hi", "hello", "hey"):
        return "Hello! Nice to meet you."
        
    # 2. Status Inquiry
    elif cleaned_input == "how are you":
        return "I'm doing great. Thanks for asking!"
        
    # 3. Identity and Name
    elif cleaned_input in ("what is your name", "who are you"):
        return "I am a Rule-Based AI Chatbot created using Python."
        
    # 4. System Time
    elif cleaned_input in ("time", "current time"):
        return f"The current system time is {utils.get_current_time()}."
        
    # 5. System Date
    elif cleaned_input in ("date", "today"):
        return f"Today's date is {utils.get_current_date()}."
        
    # 6. Help Message
    elif cleaned_input == "help":
        return get_help_message()
        
    # 7. Gratitude
    elif cleaned_input in ("thanks", "thank you"):
        return "You're welcome!"
        
    # 8. Creator Information
    elif cleaned_input == "who created you":
        return "I was created as part of an Artificial Intelligence internship project."
        
    # 9. Jokes
    elif cleaned_input == "tell me a joke":
        return random.choice(JOKES)
        
    # 10. Motivation
    elif cleaned_input == "motivate me":
        return random.choice(MOTIVATIONAL_QUOTES)
        
    # 11. Goodbye
    elif cleaned_input in ("bye", "exit", "quit"):
        return "Goodbye! Have a great day!"
        
    # 12. Fallback / Unknown Input
    else:
        return "Sorry, I don't understand that.\nPlease type 'help' to see available commands."
