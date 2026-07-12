# Rule-Based AI Chatbot

A modular, clean, and well-documented console-based rule-based AI chatbot implemented in pure Python. The project is designed for an Artificial Intelligence internship assignment, showcasing clean coding patterns, modular software design, and input sanitation without relying on external libraries or API services.

---

## 🌟 Features

- **Rule-Based Matches**: Pattern recognition for greetings, status checks, chatbot name/creator inquiries, dates, times, jokes, and motivational quotes.
- **Continuous Conversation**: Runs interactively in a CLI environment using an optimized event-driven simulation loop.
- **Input Sanitization**: Automatically trims whitespace and normalizes text casing to ensure robust rule matching.
- **Conversation Counter**: Dynamically tracks and displays turn numbers per session (e.g. `User #1:`, `Bot:`).
- **Chat History Logging**: Tracks the entire conversation session. Users can view their log of prompts and responses by typing `history`.
- **Console Controls**: Supports the `clear` command to wipe the terminal cleanly.
- **Error Handling**: Wrapped in safety blocks to capture unexpected behavior or exit interrupts (`Ctrl+C`) without crashing.

---

## 🛠️ Technologies Used

- **Language**: Python 3.8+
- **Standard Library Modules**:
  - `datetime` (system time & date querying)
  - `os` (platform-independent console clearing)
  - `random` (dynamic joke & quote selection)

No third-party packages or AI APIs (like OpenAI, Gemini, PyTorch, TensorFlow) are required.

---

## 📁 Folder Structure

```text
RuleBasedChatbot/
│
├── chatbot.py        # Main application loop and console execution engine
├── responses.py      # Predefined responses database and if-elif-else match rules
├── utils.py          # Helper functions (time, date, clear screen, clean input)
├── requirements.txt  # Project dependencies specification (Standard Library only)
└── README.md         # Professional documentation
```

---

## ⚙️ Installation & Usage

### Prerequisites
- Python 3.8 or higher installed on your computer.

### Step 1: Clone or navigate to the directory
Open your command prompt or terminal and navigate to the project directory:
```bash
cd RuleBasedChatbot
```

### Step 2: Install dependencies
Since this application only uses the Python standard library, there are no dependencies to install. However, you can verify this by checking `requirements.txt`:
```bash
cat requirements.txt
```

### Step 3: Run the Chatbot
Start the chatbot by executing:
```bash
python chatbot.py
```

---

## 💬 Example Conversation

```text
===================================
      Rule-Based AI Chatbot
===================================
Hello! I am your AI assistant.
Type 'help' to see available commands.
Type 'bye', 'exit', or 'quit' to quit.
===================================

User #1: hello
Bot: Hello! Nice to meet you.

User #2: how are you
Bot: I'm doing great. Thanks for asking!

User #3: what is your name
Bot: I am a Rule-Based AI Chatbot created using Python.

User #4: time
Bot: The current system time is 23:14:02.

User #5: date
Bot: Today's date is 2026-07-10.

User #6: motivate me
Bot: First, solve the problem. Then, write the code. - John Johnson

User #7: tell me a joke
Bot: Why do programmers wear glasses? Because they can't C#!

User #8: history

=============================================
                CHAT HISTORY
=============================================
Turn #1
  User: hello
  Bot:  Hello! Nice to meet you.
---------------------------------------------
Turn #2
  User: how are you
  Bot:  I'm doing great. Thanks for asking!
---------------------------------------------
...
=============================================

User #9: bye
Bot: Goodbye! Have a great day!
```

---

## 🚀 Future Improvements

1. **Persistent History**: Save conversation logs to a local `.txt` or `.json` file to retrieve past sessions.
2. **Regex Patterns**: Enhance command detection using Python's `re` module for substring queries (e.g. recognizing "hey chatbot" or "give me a joke").
3. **Web Interface**: Build a web-based chat UI using Python frameworks like Streamlit or Flask.
