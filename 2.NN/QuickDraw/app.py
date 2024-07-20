"""En simpel app til at tegne og gætte hvad der er tegnet"""

import time
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import cv2
import numpy as np
from data._data_static import TEGNINGER, dsize

# Husk at ændre til den model du gerne vil teste
MODEL_TIL_TEST = "2024-07-20-11-08-20-snwc.pth"

@st.cache_resource
def get_model(path):
    """Læs modellen fra den gemte fil"""
    model = torch.jit.load(path)

    # Sæt modellen til evaluation mode
    model.eval()
    return model

def predict(model: torch.jit.ScriptModule, tegning: np.ndarray):
    """Få modellen til at forudsige hvad der er tegnet"""
    # Forbered tegningen til klassificering
    if tegning is not None:
        tegning_til_klassificering = cv2.cvtColor(tegning.astype(np.uint8),cv2.COLOR_BGR2GRAY)
        tegning_til_klassificering = cv2.bitwise_not(tegning_til_klassificering)
        tegning_til_klassificering = cv2.resize(tegning_til_klassificering, dsize = dsize, interpolation = cv2.INTER_AREA)

    # Forudsig hvad der er tegnet, hvis der er en tegning
    if tegning_til_klassificering.max() > 0:
        start_time = time.time()
        image = torch.tensor(tegning_til_klassificering).float()
        image = image.to(next(model.parameters()).device)
        image = image.reshape(-1, 1, 28, 28)
        
        with torch.no_grad():
            y_hat_prob = model(image)
            y_hat_prob = torch.nn.functional.softmax(y_hat_prob, dim=1)
            y_hat = torch.argmax(y_hat_prob, dim=1)

        y_hat = y_hat.detach().numpy()[0]
        y_hat_prob = y_hat_prob[0].detach().numpy()

        prob = round(y_hat_prob[y_hat] * 100, 2)
        execution_time = time.time() - start_time

        # Returner forudsigelsen i appen
        st.session_state.messages.append({
            "type": "assistant",
            "message": f"Jeg er {prob}% sikker på at det er *{TEGNINGER[y_hat]}*. Det fandt jeg ud af på {execution_time:.3f} sekunder"
        })

def write_tegn_og_gaet(model: torch.jit.ScriptModule):
    # To colonner: Første til tegning, anden til api forbindelser
    col1, _, col2 = st.columns([1, 0.05, 1])

    # Tegneboks
    with col1:
        streg = st.slider("Stregtykkelse", 25, 50, 30, 1)
        tegning = st_canvas(
            fill_color="rgba(255, 165, 0, 0)",  # Fixed fill color with some opacity
            stroke_width= streg,
            stroke_color="#000000",
            background_color="#FFFFFF",
            background_image = None,
            update_streamlit=True,
            height = 750,
            width = 750,
            drawing_mode="freedraw",
            key="canvas",
        )
        
    with col2:
        st.write("")
        st.write("")
        # Når der er tegnet, sendes tegningen til alle holdene
        predict(model, tegning.image_data)
        for message in st.session_state.messages:
            with st.chat_message(message["type"]):
                st.markdown(message["message"])
# Indstillinger
st.set_page_config(page_title="UNFML24 Tegn og Gæt", page_icon=".streamlit/unflogo.svg", layout="wide")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Title på side
col1, _, col2 = st.columns([10, 1, 2])
col1.title("Tegn og gæt konkurrence")
col1.subheader(f"Tester model: {MODEL_TIL_TEST}")
reload = col1.button("Genload model")
col2.image('.streamlit/unflogo.svg', width=160)

st.divider()

model = get_model(f"saved_models/{MODEL_TIL_TEST}")
write_tegn_og_gaet(model=model)
_, col3 = st.columns([1, 0.1])
reset_button = col3.button("Nulstil chat", type="secondary")

if reset_button:
    st.session_state.messages = []

if reload:
    st.cache_resource.clear()
st.caption("UNFML24 Tegn og Gæt konkurrence - bygget af UNFML24 with :heart:")