import os
import gradio as gr
from huggingface_hub import InferenceClient

HF_TOKEN = os.getenv("HF_TOKEN")

EMISSIONS_FACTORS = {
    "transportation": {
        "car": 2.3,
        "bus": 0.1,
        "train": 0.04,
        "plane": 0.25,
    },
    "food": {
        "meat": 6.0,
        "vegetarian": 1.5,
        "vegan": 1.0,
    }
}

def calculate_footprint(car_km, bus_km, train_km, air_km, meat_meals, vegetarian_meals, vegan_meals):
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
        "driving100km": round(total_emissions / 230),
    }

    return total_emissions, stats

def respond(
    message,
    history: list[dict[str, str]],
    system_message,
    car_km,
    bus_km,
    train_km,
    air_km,
    meat_meals,
    vegetarian_meals,
    vegan_meals,
):
    client = InferenceClient(token=HF_TOKEN, model="openai/gpt-oss-20b")

    footprint, stats = calculate_footprint(
        car_km, bus_km, train_km, air_km,
        meat_meals, vegetarian_meals, vegan_meals
    )

    custom_prompt = f"""
This user’s estimated weekly footprint is **{footprint:.1f} kg CO2**.
That’s equivalent to planting about {stats['trees']} trees 🌳 or taking {stats['flights']} short flights ✈️.
Their breakdown includes both transportation and food habits.  
Your job is to guide them with practical, encouraging suggestions to lower this footprint.
{system_message}
"""

    max_tokens = 512
    temperature = 0.7
    top_p = 0.95

    messages = [{"role": "system", "content": custom_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": message})

    response = ""
    for message in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        choices = message.choices
        token = ""
        if len(choices) and choices[0].delta.content:
            token = choices[0].delta.content
        response += token
        yield response

system_prompt = """
You are Sustainable.ai, a friendly, encouraging, and knowledgeable AI assistant...
(omit full content for brevity – use your full prompt here)
"""

with gr.Blocks(css="""
    body {
        background: linear-gradient(135deg, #e0f7fa, #f1f8e9);
    }
    .section-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .title-text {
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        color: #2e7d32;
    }
    .subtitle-text {
        text-align: center;
        font-size: 16px;
        color: #555;
        margin-bottom: 20px;
    }
""") as demo:

    with gr.Column():
        gr.HTML("<div class='title-text'>🌍 Eco Wise AI</div>")
        gr.HTML("<div class='subtitle-text'>Track your weekly habits and chat with your personal sustainability coach 🌱</div>")

    with gr.Group(elem_classes="section-card"):
        gr.Markdown("### 🚗 Transportation (per week)")
        with gr.Row():
            car_input = gr.Number(label="🚘 Car Travel (km)", value=0)
            bus_input = gr.Number(label="🚌 Bus Travel (km)", value=0)
        with gr.Row():
            train_input = gr.Number(label="🚆 Train Travel (km)", value=0)
            air_input = gr.Number(label="✈️ Air Travel (km/month)", value=0)

  
    with gr.Group(elem_classes="section-card"):
        gr.Markdown("### 🍽️ Food Habits (per week)")
        with gr.Row():
            meat_input = gr.Number(label="🥩 Meat Meals", value=0)
            vegetarian_input = gr.Number(label="🥗 Vegetarian Meals", value=0)
            vegan_input = gr.Number(label="🌱 Vegan Meals", value=0)

    
    chatbot = gr.ChatInterface(
        respond,
        type="messages",
        additional_inputs=[
            gr.Textbox(value=system_prompt, visible=False),
            car_input,
            bus_input,
            train_input,
            air_input,
            meat_input,
            vegetarian_input,
            vegan_input,
        ],
    )

if __name__ == "__main__":
    demo.launch()
