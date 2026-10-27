
import gradio as gr
import tensorflow as tf
import numpy as np
from pathlib import Path

MODEL_PATH = Path("/mnt/c/vs code/deep fake image detection/models/deepfake_detector_final.keras")
THRESHOLD_PATH = Path("/mnt/c/vs code/deep fake image detection/models/threshold.txt")

model = tf.keras.models.load_model(MODEL_PATH)

with open(THRESHOLD_PATH, "r") as f:
    THRESHOLD = float(f.read().strip())


def detect_deepfake(image):
    if image is None:
        return (
            "WAITING FOR IMAGE",
            "Upload an image to begin analysis.",
            "0.00%",
            "0.00%"
        )

    img = tf.convert_to_tensor(image)

    if img.shape[-1] == 4:
        img = img[:, :, :3]

    img = tf.image.resize(img, (224, 224))
    img = tf.expand_dims(img, axis=0)

    probability_real = float(
        model.predict(img, verbose=0)[0][0]
    )

    probability_fake = 1.0 - probability_real

    if probability_real >= THRESHOLD:
        result = "REAL"
        message = "The image is classified as REAL."
    else:
        result = "FAKE"
        message = "The image is classified as FAKE."

    return (
        result,
        message,
        f"{probability_real * 100:.2f}%",
        f"{probability_fake * 100:.2f}%"
    )


css = """
body {
    background:
        radial-gradient(circle at 20% 20%, #172554 0%, transparent 35%),
        radial-gradient(circle at 80% 80%, #3f0f46 0%, transparent 35%),
        #05070d;
    color: white;
}

.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
    background: transparent !important;
}

.hero {
    text-align: center;
    padding: 45px 20px 30px;
}

.hero-title {
    font-size: 48px;
    font-weight: 900;
    letter-spacing: 4px;
    margin-bottom: 8px;
}

.hero-subtitle {
    color: #9ca3af;
    font-size: 17px;
    letter-spacing: 1px;
}

.glass {
    background: rgba(255,255,255,0.055);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 24px;
    padding: 25px;
}

.footer {
    text-align: center;
    color: #6b7280;
    font-size: 13px;
    padding: 25px;
}
"""

with gr.Blocks(title="DeepFake Detector", css=css) as demo:

    gr.HTML("""
    <div class="hero">
        <div class="hero-title">DEEPFAKE DETECTOR</div>
        <div class="hero-subtitle">
            AI-POWERED IMAGE AUTHENTICITY ANALYSIS
        </div>
    </div>
    """)

    with gr.Row():

        with gr.Column(scale=1, elem_classes="glass"):
            gr.Markdown("### 📤 Upload Image")

            image_input = gr.Image(
                type="numpy",
                label="Upload a face image",
                height=420
            )

            analyze_button = gr.Button(
                "🔍 ANALYZE IMAGE",
                variant="primary",
                size="lg"
            )

        with gr.Column(scale=1, elem_classes="glass"):
            gr.Markdown("### 🧠 Analysis Result")

            result = gr.Textbox(
                label="Detection",
                value="WAITING FOR IMAGE",
                text_align="center"
            )

            message = gr.Textbox(
                label="Analysis",
                value="Upload an image to begin analysis.",
                text_align="center"
            )

            with gr.Row():
                real_probability = gr.Textbox(
                    label="REAL PROBABILITY",
                    value="0.00%",
                    text_align="center"
                )

                fake_probability = gr.Textbox(
                    label="FAKE PROBABILITY",
                    value="0.00%",
                    text_align="center"
                )

    gr.HTML("""
    <div class="footer">
        EfficientNetB0 • Deep Learning • Binary Image Classification<br>
        Validation-selected threshold: 0.20
    </div>
    """)

    analyze_button.click(
        fn=detect_deepfake,
        inputs=image_input,
        outputs=[
            result,
            message,
            real_probability,
            fake_probability
        ]
    )


if __name__ == "__main__":
    demo.launch(
        share=True,
        theme=gr.themes.Base(
            primary_hue="blue",
            neutral_hue="slate"
        )
    )
