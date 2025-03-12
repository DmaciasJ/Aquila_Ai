import speech_recognition as sr
import pyttsx3
import time
import threading
import keyboard
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# Conversation logging template
template = """
Answer the question below.

Here is the conversation history: {context}

Question: {question}

Answer:
"""

# Load the AI model
model = OllamaLLM(model="llama3.1")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def log_conversation(user_input, response):
    """Log conversation to file."""
    with open("conversation_log.txt", "a") as log_file:
        log_file.write(f"User: {user_input}\nAquila: {response}\n\n")

def load_conversation_history():
    """Load past conversation history from the log file."""
    try:
        with open("conversation_log.txt", "r") as log_file:
            return log_file.read()
    except FileNotFoundError:
        return ""

def handle_conversation(engine, context):
    """Handle the conversation and response."""
    recognizer = sr.Recognizer()

    # Set properties for a more human-sounding voice
    voices = engine.getProperty('voices')
    for voice in voices:
        if "en_US" in voice.id:  # Select American English voice
            engine.setProperty('voice', voice.id)
            break
    engine.setProperty('rate', 150)  # Slightly faster speech rate for a natural sound
    engine.setProperty('volume', 1.0)  # Volume level (0.0 to 1.0)
    engine.setProperty('pitch', 75)  # Control pitch to sound more natural

    while True:
        with sr.Microphone() as source:
            print("Listening...")
            audio = recognizer.listen(source)
            try:
                user_input = recognizer.recognize_google(audio)
                print(f"You: {user_input}")
                
                if user_input.lower() == "exit":
                    print("Exiting the conversation.")
                    break
                else:
                    # Format input for the model with conversation history
                    formatted_input = prompt.format(context=context, question=user_input)
                    result = model.invoke(formatted_input)
                    print("Aquila:", result)

                    # Output response and update history
                    engine.say(result)
                    engine.runAndWait()
                    context += f"\nUser: {user_input}\nAquila: {result}"
                    log_conversation(user_input, result)

            except sr.UnknownValueError:
                time.sleep(1)  # Add a delay of 1 second
                print("Sorry, I did not understand that.")
                engine.say("Sorry, I did not understand that.")
                engine.runAndWait()
            except sr.RequestError as e:
                print(f"Could not request results; {e}")
                engine.say("Could not request results.")
                engine.runAndWait()

def listen_for_activation():
    """Listen continuously for 'Hey Aquila' command."""
    recognizer = sr.Recognizer()

    while True:
        with sr.Microphone() as source:
            print("Listening for activation...")
            audio = recognizer.listen(source)
            try:
                # Recognize the wake word "Hey Aquila"
                user_input = recognizer.recognize_google(audio)
                print(f"Listening for activation: {user_input}")
                if "aquila" in user_input.lower():
                    print("Aquila activated!")
                    return True
            except sr.UnknownValueError:
                pass
            except sr.RequestError as e:
                print(f"Could not request results; {e}")

def run_in_background():
    """Run Aquila in the background, awaiting activation."""
    engine = pyttsx3.init()

    context = load_conversation_history()  # Load previous conversation context

    # Start listening for activation
    print("Aquila is running in the background. Say 'Aquila ...' to activate.")
    while True:
        if keyboard.is_pressed('esc'):  # Press ESC to stop talking
            print("Speech paused. Aquila will stop talking.")
            continue

        if listen_for_activation():
            handle_conversation(engine, context)  # Start conversation once activated

if __name__ == "__main__":
    run_in_background()