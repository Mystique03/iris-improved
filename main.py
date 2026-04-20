import warnings
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")

from iris.model import load_and_train, predict
from iris.voice import listen, speak
from iris.llm import get_treatments, get_diet_chart

SYMPTOM_KEYWORDS = ("symptom", "predict", "diagnose", "sick", "feeling", "pain", "unwell", "disease")


def handle_symptoms(model, symptom_columns, le):
    symptoms_text = listen("Please describe your symptoms.")
    if not symptoms_text:
        speak("I didn't catch that. Please try again.")
        return

    disease = predict(model, symptom_columns, symptoms_text, le)
    speak(f"Based on your symptoms, you may have {disease}. Please consult a doctor for confirmation.")

    speak("Fetching treatment options, please wait.")
    treatments = get_treatments(disease)
    speak(f"Here are the treatment options for {disease}.")
    speak(treatments)

    reply = listen("Would you like a diet chart as well? Say yes or no.")
    if reply and "yes" in reply.lower():
        speak("Preparing your diet chart, please wait.")
        chart = get_diet_chart(disease)
        speak(f"Here is the recommended diet for {disease}.")
        speak(chart)
        speak("Please consult a healthcare professional for a personalised plan.")


def handle_diet():
    concern = listen("What is your dietary concern or condition?")
    if not concern:
        speak("I didn't catch that. Please try again.")
        return
    speak("Preparing your diet chart, please wait.")
    chart = get_diet_chart(concern)
    speak(f"Here is the recommended diet for {concern}.")
    speak(chart)
    speak("Please consult a healthcare professional for a personalised plan.")


def conversation(model, symptom_columns, le, name):
    while True:
        intent = listen(f"Hello {name}. How can I help you? Say 'symptoms' to predict a disease or 'diet' for a diet chart.")
        if not intent:
            speak("I didn't catch that. Please try again.")
            continue

        lower = intent.lower()

        if any(w in lower for w in ("stop", "exit", "bye", "goodbye")):
            speak("Goodbye, have a good day.")
            return False

        if "diet" in lower:
            handle_diet()
        elif any(w in lower for w in SYMPTOM_KEYWORDS):
            handle_symptoms(model, symptom_columns, le)
        else:
            speak("I can help with symptom prediction or diet charts. Please say 'symptoms' or 'diet chart'.")
            continue

        reply = listen("Is there anything else I can help you with? Say yes or no.")
        if not reply or any(w in reply.lower() for w in ("no", "stop", "exit", "bye")):
            speak("Goodbye, have a good day.")
            return False

        if "yes" not in reply.lower():
            speak("Goodbye, have a good day.")
            return False


def run():
    print("Loading model...")
    model, symptom_columns, le = load_and_train()
    print("Model ready. Say 'Hello Doctor' to begin.")

    while True:
        text = listen()
        if not text:
            continue

        lower = text.lower()

        if any(w in lower for w in ("stop", "exit", "bye")):
            speak("Goodbye, have a good day.")
            break

        if "hello doctor" not in lower:
            speak("Say 'Hello Doctor' to begin.")
            continue

        name = listen("Hello! What is your name?")
        if not name:
            name = "there"
        else:
            name = name.strip().split()[0].capitalize()

        keep_going = conversation(model, symptom_columns, le, name)
        if not keep_going:
            break


if __name__ == "__main__":
    run()
