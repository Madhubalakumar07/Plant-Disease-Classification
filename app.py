import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from model import Model


# -------------------------
# Page configuration
# -------------------------
st.set_page_config(
    page_title="LeafLens | Plant Health",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --ink: #16352b;
        --muted: #6b8178;
        --green: #2f8f61;
        --green-dark: #1d6845;
        --mint: #eaf7ef;
        --line: #dcebe2;
        --surface: rgba(255, 255, 255, 0.92);
    }

    .stApp {
        background:
            radial-gradient(circle at 8% 0%, rgba(171, 224, 186, .42), transparent 29rem),
            radial-gradient(circle at 94% 14%, rgba(255, 226, 164, .23), transparent 25rem),
            #f7fbf8;
        color: var(--ink);
        font-family: 'DM Sans', sans-serif;
    }

    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"] { right: 1rem; }
    .block-container { max-width: 1180px; padding: 2.5rem 2rem 3rem; }

    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; color: var(--ink) !important; }
    .eyebrow {
        color: var(--green-dark); font-size: .78rem; font-weight: 700;
        letter-spacing: .16em; text-transform: uppercase; margin-bottom: .5rem;
    }
    .hero-title { font-family: 'Space Grotesk', sans-serif; font-size: clamp(2.4rem, 5vw, 4.7rem);
        line-height: .98; letter-spacing: -.06em; color: var(--ink); margin: 0; }
    .hero-title span { color: var(--green); }
    .hero-copy { color: var(--muted); font-size: 1.08rem; line-height: 1.65; max-width: 620px; margin: 1rem 0 0; }
    .hero-badge { background: var(--mint); border: 1px solid #cdebd7; border-radius: 999px;
        color: var(--green-dark); display: inline-block; font-size: .82rem; font-weight: 600;
        padding: .45rem .8rem; margin-top: 1.35rem; }
    .panel { background: var(--surface); border: 1px solid var(--line); border-radius: 24px;
        box-shadow: 0 15px 45px rgba(39, 89, 62, .08); padding: 1.35rem; }
    .panel-title { color: var(--ink); font-family: 'Space Grotesk', sans-serif; font-size: 1.05rem;
        font-weight: 700; margin-bottom: .2rem; }
    .panel-caption { color: var(--muted); font-size: .88rem; margin-bottom: 1rem; }
    [data-testid="stFileUploader"] { background: #f7fcf8; border: 1.5px dashed #9bceb0;
        border-radius: 18px; padding: .5rem; }
    [data-testid="stFileUploaderDropzone"] { background: transparent; }
    [data-testid="stFileUploader"] small { color: var(--muted); }
    .result-card { background: linear-gradient(135deg, #eaf8ef 0%, #f8fcf9 100%);
        border: 1px solid #c8e7d2; border-radius: 20px; padding: 1.35rem; margin-top: 1rem; }
    .result-label { color: var(--green-dark); font-size: .76rem; font-weight: 700;
        letter-spacing: .12em; text-transform: uppercase; }
    .result-name { color: var(--ink); font-family: 'Space Grotesk', sans-serif; font-size: 1.6rem;
        font-weight: 700; line-height: 1.15; margin-top: .35rem; }
    .metric { background: #fff; border: 1px solid var(--line); border-radius: 16px; padding: .9rem 1rem; }
    .metric-label { color: var(--muted); font-size: .75rem; text-transform: uppercase; letter-spacing: .08em; }
    .metric-value { color: var(--ink); font-family: 'Space Grotesk', sans-serif; font-size: 1.25rem;
        font-weight: 700; margin-top: .25rem; }
    .empty-state { align-items: center; background: #fbfefc; border: 1px dashed var(--line);
        border-radius: 18px; color: var(--muted); display: flex; justify-content: center;
        min-height: 300px; text-align: center; }
    .footer { color: #89a095; font-size: .78rem; padding-top: 2.5rem; text-align: center; }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------------
# Device
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------
# Load saved model
# -------------------------
@st.cache_resource
def load_model():
    checkpoint = torch.load("Model/plant_disease_model.pth", map_location=device)
    loaded_model = Model(checkpoint["num_classes"])
    loaded_model.load_state_dict(checkpoint["model_state_dict"])
    loaded_model.to(device)
    loaded_model.eval()
    return loaded_model, checkpoint["class_names"]


model, class_names = load_model()


def format_label(label):
    """Make dataset-style labels easier to read in the interface."""
    return label.replace("___", " · ").replace("__", " · ").replace("_", " ").strip()


# -------------------------
# Streamlit UI
# -------------------------
st.markdown('<div class="eyebrow">AI-powered plant care</div>', unsafe_allow_html=True)
st.markdown(
    '<h1 class="hero-title">See what your <span>leaves</span><br>are telling you.</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="hero-copy">Upload a clear photo of a plant leaf and LeafLens will identify '
    'potential diseases in seconds, helping you take action sooner.</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero-badge">✦ Fast, private &nbsp;·&nbsp; Runs locally on your device</div>',
    unsafe_allow_html=True,
)

st.write("")
upload_col, info_col = st.columns([1.35, 1], gap="large")

with upload_col:
    st.markdown(
        '<div class="panel"><div class="panel-title">Start with a leaf photo</div>'
        '<div class="panel-caption">For best results, use a well-lit image with one leaf in focus.</div>',
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "Drop an image here, or browse your files",
        type=["jpg", "jpeg", "png"],
        label_visibility="visible",
    )
    st.markdown("</div>", unsafe_allow_html=True)

with info_col:
    st.markdown(
        '<div class="panel"><div class="panel-title">How it works</div>'
        '<div class="panel-caption">Three simple steps to healthier plants.</div>'
        '<div style="color:#2f8f61;font-size:1.35rem;font-weight:700">01</div>'
        '<div style="font-weight:600;margin:.15rem 0 .65rem">Upload</div>'
        '<div style="color:#6b8178;font-size:.88rem;margin-bottom:1rem">Share a clear JPG or PNG photo.</div>'
        '<div style="color:#2f8f61;font-size:1.35rem;font-weight:700">02</div>'
        '<div style="font-weight:600;margin:.15rem 0 .65rem">Analyze</div>'
        '<div style="color:#6b8178;font-size:.88rem;margin-bottom:1rem">Our trained model scans visual patterns.</div>'
        '<div style="color:#2f8f61;font-size:1.35rem;font-weight:700">03</div>'
        '<div style="font-weight:600;margin:.15rem 0 .65rem">Act</div>'
        '<div style="color:#6b8178;font-size:.88rem">Use the result as a quick first check.</div></div>',
        unsafe_allow_html=True,
    )


if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.write("")
    preview_col, result_col = st.columns([1, 1], gap="large")

    with preview_col:
        st.markdown(
            '<div class="panel"><div class="panel-title">Your leaf</div>'
            '<div class="panel-caption">Ready for analysis</div>',
            unsafe_allow_html=True,
        )
        st.image(image, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with result_col:
        with st.spinner("Reading leaf patterns..."):
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ])
            image_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                probabilities = torch.softmax(model(image_tensor), dim=1)
                confidence, predicted_class = torch.max(probabilities, dim=1)

        predicted_label = format_label(class_names[predicted_class.item()])
        confidence_percent = confidence.item() * 100
        confidence_tone = "High confidence" if confidence_percent >= 75 else "Review suggested"

        st.markdown(
            f'<div class="panel"><div class="panel-title">Analysis complete</div>'
            f'<div class="panel-caption">Most likely match from the visual scan</div>'
            f'<div class="result-card"><div class="result-label">Detected condition</div>'
            f'<div class="result-name">{predicted_label}</div></div>',
            unsafe_allow_html=True,
        )
        metric_col, device_col = st.columns(2, gap="small")
        with metric_col:
            st.markdown(
                f'<div class="metric"><div class="metric-label">Confidence</div>'
                f'<div class="metric-value">{confidence_percent:.1f}%</div></div>',
                unsafe_allow_html=True,
            )
        with device_col:
            st.markdown(
                f'<div class="metric"><div class="metric-label">Status</div>'
                f'<div class="metric-value" style="font-size:1rem">{confidence_tone}</div></div>',
                unsafe_allow_html=True,
            )
        st.progress(confidence.item(), text="Prediction confidence")
        st.caption(f"Processed locally on {str(device).upper()}. Use this result as a screening aid.")
        st.markdown("</div>", unsafe_allow_html=True)
else:
    st.write("")
    st.markdown(
        '<div class="empty-state"><div><div style="font-size:2rem">🍃</div>'
        '<div style="font-weight:600;color:#557167;margin-top:.5rem">Your analysis will appear here</div>'
        '<div style="font-size:.85rem;margin-top:.3rem">Upload a leaf image to get started</div></div></div>',
        unsafe_allow_html=True,
    )

st.markdown('<div class="footer">LeafLens · Plant health, made easier</div>', unsafe_allow_html=True)