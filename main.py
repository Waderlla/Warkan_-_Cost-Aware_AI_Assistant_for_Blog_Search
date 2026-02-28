import gradio as gr
from transformers import pipeline

# Mały, lekki model instrukcyjny (działa na CPU)
# Jakość: "OK do prostych odpowiedzi", stabilność: wysoka
MODEL_ID = "google/flan-t5-small"

gen = pipeline(
    "text2text-generation",
    model=MODEL_ID,
)

MAX_TURNS = 6
MAX_INPUT_CHARS = 800
MAX_NEW_TOKENS = 128

SYSTEM_HINT = (
    "Jesteś asystentem bloga Waderlla. Odpowiadaj krótko, konkretnie i po polsku. "
    "Jeśli nie wiesz, powiedz wprost i zaproponuj co użytkownik może sprawdzić na blogu."
)

def chat(message, history):
    message = (message or "").strip()
    if not message:
        return "Napisz wiadomość."

    message = message[:MAX_INPUT_CHARS]
    history = history[-MAX_TURNS:] if history else []

    # Budujemy krótki kontekst z historii (żeby nie rosło w nieskończoność)
    context_lines = [SYSTEM_HINT]
    for u, a in history:
        if u:
            context_lines.append(f"Użytkownik: {str(u)[:MAX_INPUT_CHARS]}")
        if a:
            context_lines.append(f"Asystent: {str(a)[:MAX_INPUT_CHARS]}")
    context_lines.append(f"Użytkownik: {message}")
    context_lines.append("Asystent:")

    prompt = "\n".join(context_lines)

    out = gen(prompt, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)[0]["generated_text"]
    # flan-t5 potrafi zwrócić cały tekst; wycinamy tylko końcówkę po "Asystent:"
    if "Asystent:" in out:
        out = out.split("Asystent:", 1)[-1].strip()

    return out or "Nie jestem pewien. Spróbuj doprecyzować pytanie."

demo = gr.ChatInterface(
    fn=chat,
    title="Warkan AI",
    description="Asystent bloga (darmowy, CPU).",
)

demo.launch()
