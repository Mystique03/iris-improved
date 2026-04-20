import pyttsx3
import speech_recognition as sr

_recognizer = sr.Recognizer()


def speak(text: str) -> None:
    print(f"Iris: {text}")
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    engine.stop()


def listen(prompt: str | None = None) -> str | None:
    if prompt:
        speak(prompt)
    try:
        with sr.Microphone() as source:
            _recognizer.adjust_for_ambient_noise(source, duration=0.5)
            print("Listening...")
            audio = _recognizer.listen(source, timeout=10, phrase_time_limit=15)
    except sr.WaitTimeoutError:
        return None
    try:
        text = _recognizer.recognize_google(audio)
        print(f"Heard: {text}")
        return text
    except sr.UnknownValueError:
        return None
    except sr.RequestError as e:
        print(f"STT error: {e}")
        return None
