# Iris — Improved

Iris is a voice-interactive medical assistant that listens to your symptoms, predicts a likely disease using a ML model (xgboost), and provides treatment options and diet recommendations powered by an LLM.

---

## Features

- **Voice interaction** — speak naturally; Iris listens and responds using text-to-speech (new)
- **Symptom-based disease prediction** — XGBoost classifier trained on a labelled symptom dataset (new)
- **Hyperparameter tuning** — GridSearchCV runs once and saves best params to `data/best_params.json`; subsequent loads skip the search
- **Treatment recommendations** — fetched from Groq based on predicted disease (new)
- **Diet chart generation** — personalised food recommendations for a given condition
- **Wake-word activation** — session starts only when you say "Hello Doctor"

---

## Project Structure

```
iris-voice-assistant/
├── data/
│   ├── Training.csv          # Training dataset (symptoms + prognosis)
│   ├── Testing.csv           # Testing dataset
│   ├── best_params.json      # grid search results
│   └── ...                   # Supporting CSVs
├── iris/
│   ├── model.py              # XGBoost training, grid search, prediction
│   ├── voice.py              # Speech recognition (STT) + text-to-speech (TTS)
│   └── llm.py                # Groq LLM calls for treatments and diet charts
├── main.py                   # Entry point and conversation flow
├── pyproject.toml
└── .env                      # API keys (not committed)
```

---

## Requirements

- Python 3.11+
- A microphone
- [uv](https://github.com/astral-sh/uv) (package manager)
- A free [Groq API key](https://console.groq.com)

---

## Setup

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd iris-voice-assistant
   ```

2. **Install dependencies**
   ```bash
   uv sync
   ```

3. **Create a `.env` file** in the project root:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

4. **Run the assistant**
   ```bash
   uv run main.py
   ```

---

## How to Use

1. Run the app and wait for the model to load
2. Say **"Hello Doctor"** to start a session
3. Tell Iris your name when prompted
4. Choose what you need:
   - Say **"symptoms"** or **"I feel sick"** → describe your symptoms → get a diagnosis + treatments
   - Say **"diet chart"** → describe your condition → get food recommendations
5. After each response, Iris asks if you need anything else
6. Say **"no"**, **"bye"**, or **"exit"** to end the session

---

## ML Model Details

| Property | Value |
|---|---|
| Algorithm | XGBoost (`XGBClassifier`) |
| Tuning | GridSearchCV (5-fold CV) |
| Parameters searched | `n_estimators`, `max_depth`, `learning_rate` |
| Label encoding | `sklearn.LabelEncoder` |
| Train/test split | 67% / 33% |

Best hyperparameters are saved to `data/best_params.json` after the first run. Delete this file to re-run the grid search.

---

## Tech Stack

| Component | Library |
|---|---|
| ML model | `xgboost`, `scikit-learn` |
| Data | `pandas`, `numpy` |
| Speech recognition | `SpeechRecognition`, `PyAudio` |
| Text-to-speech | `pyttsx3` |
| LLM | Groq API (`llama-3.3-70b-versatile`) |
| Env config | `python-dotenv` |
