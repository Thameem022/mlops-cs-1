import os
import gradio as gr
from huggingface_hub import InferenceClient

# --- Emissions factors --------------------------------------------------------
EMISSIONS_FACTORS = {
    "transportation": {"car": 2.3, "bus": 0.1, "train": 0.04, "plane": 0.25},
    "food": {"meat": 6.0, "vegetarian": 1.5, "vegan": 1.0},
}

def calculate_footprint(car_km, bus_km, train_km, air_km,
                        meat_meals, vegetarian_meals, vegan_meals):
    transport_emissions = (
        car_km * EMISSIONS_FACTORS["transportation"]["car"] +
        bus_km * EMISSIONS_FACTORS["transportation"]["bus"] +
        train_km * EMISSIONS_FACTORS["transportation"]["train"] +
        air_km * EMISSIONS_FACTORS["transportation"]["plane"]
    )
    food_emissions = (
        meat_meals * EMISSIONS_FACTORS["food"]["meat"] +
        vegetarian_meals * EMISSIONS_FACTORS["food"]["vegetarian"] +
        vegan_meals * EMISSIONS_FACTORS["food"]["vegan"]
    )
    total_emissions = transport_emissions + food_emissions
    stats = {
        "trees": round(total_emissions / 21),      
        "flights": round(total_emissions / 500),     
        "driving100km": round(total_emissions / 230) 
    }
    return total_emissions, stats

# --- Default system prompt ----------------------------------------------------
DEFAULT_SYSTEM_PROMPT = """
You are Sustainable.ai, a friendly, encouraging, and knowledgeable AI assistant.
Always provide practical sustainability suggestions that are easy to adopt,
while keeping a supportive and positive tone. Prefer actionable steps over theory.
Reasoning: medium
"""

# --- Chat callback ------------------------------------------------------------
def respond(
    message,
    history: list[dict[str, str]],
    hf_token_ui,          # from password textbox (optional)
    system_message,       # from textbox
    car_km,
    bus_km,
    train_km,
    air_km,
    meat_meals,
    vegetarian_meals,
    vegan_meals,
):
    """
    Streams a response from openai/gpt-oss-20b via Hugging Face Inference API.
    Token priority: UI textbox > HF_TOKEN env var.
    """
    # Resolve token from UI or env
    token = (hf_token_ui or "").strip() or (os.getenv("HF_TOKEN") or "").strip()
    if not token:
        yield "⚠️ Please provide a valid Hugging Face token in the 'HF Token' box or set HF_TOKEN in the environment."
        return

    # Correct, namespaced repo id
    model_id = "openai/gpt-oss-20b"

    # Build client
    try:
        client = InferenceClient(model=model_id, token=token)
    except Exception as e:
        yield f"Failed to initialize InferenceClient: {e}"
        return

    # Compute personalized footprint summary
    footprint, stats = calculate_footprint(
        car_km, bus_km, train_km, air_km,
        meat_meals, vegetarian_meals, vegan_meals
    )

    custom_prompt = (
        f"This user’s estimated weekly footprint is **{footprint:.1f} kg CO2**.\n"
        f"That’s roughly planting {stats['trees']} trees 🌳 or taking {stats['flights']} short flights ✈️.\n"
        f"Breakdown includes transportation and food choices.\n"
        f"Your job is to give practical, friendly suggestions to lower this footprint.\n"
        f"{system_message}"
    )

    # Construct messages in OpenAI-style format; providers map this to the model's chat template.
    messages = [{"role": "system", "content": custom_prompt}]
    messages.extend(history or [])
    messages.append({"role": "user", "content": message})

    # Stream from HF Inference API
    try:
        response = ""
        for chunk in client.chat_completion(
            messages,
            max_tokens=3000,
            temperature=0.7,
            top_p=0.95,
            stream=True,
        ):
            try:
                # Some providers return choices[0].delta.content during streaming
                if chunk.choices and getattr(chunk.choices[0], "delta", None):
                    token_piece = chunk.choices[0].delta.content or ""
                else:
                    # Fallback: some providers may use 'message' at the end
                    token_piece = getattr(chunk, "message", {}).get("content", "") or ""
            except Exception:
                token_piece = ""

            if token_piece:
                response += token_piece
                yield response
    except Exception as e:
        # Common causes: 401 (bad token), 404 (wrong repo id), provider downtime
        yield f"Inference error with '{model_id}': {e}\n"
        return

# --- UI -----------------------------------------------------------------------
demo = gr.ChatInterface(
    fn=respond,
    type="messages",  # fixes 'tuples' deprecation warning
    additional_inputs=[
        gr.Textbox(label="HF Token (prefer env var HF_TOKEN)", type="password", placeholder="hf_..."),
        gr.Textbox(value=DEFAULT_SYSTEM_PROMPT, label="System Prompt"),
        gr.Slider(0, 500, value=50, step=10, label="Car km/week"),
        gr.Slider(0, 500, value=20, step=10, label="Bus km/week"),
        gr.Slider(0, 500, value=20, step=10, label="Train km/week"),
        gr.Slider(0, 5000, value=200, step=50, label="Air km/week"),
        gr.Slider(0, 21, value=7, step=1, label="Meat meals/week"),
        gr.Slider(0, 21, value=7, step=1, label="Vegetarian meals/week"),
        gr.Slider(0, 21, value=7, step=1, label="Vegan meals/week"),
    ],
    title="🌱 Sustainable.ai (gpt-oss-20b)",
    description=(
        "Chat with an AI that helps you understand and reduce your carbon footprint. "
        "Provide a Hugging Face token in the UI or via HF_TOKEN. Uses openai/gpt-oss-20b."
    ),
)

if __name__ == "__main__":
    demo.launch()