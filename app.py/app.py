import os
import tempfile
from collections import Counter

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Pest Detection and Population Advisory System",
    page_icon="🌾",
    layout="wide",
)

st.title("🌾 Pest Detection and Population Advisory System")
st.markdown(
    """
This application detects crop pests using a trained **YOLOv8 model** and provides
**severity-based advisory suggestions** for farmers using both **natural** and **chemical** methods.
"""
)


# =========================================================
# MODEL LOADING
# =========================================================
@st.cache_resource
def load_model(model_path: str):
    return YOLO(model_path)


DEFAULT_MODEL_PATH = "yolov8_pest_model.pt"


# =========================================================
# PEST KNOWLEDGE BASE
# =========================================================
PEST_INFO = {
    "aphid": {
        "display_name": "Aphid",
        "description": (
            "Aphids are small sap-sucking insects that weaken plants by feeding on tender "
            "leaves and stems. They can also cause leaf curling, yellowing, and transmission "
            "of plant diseases."
        ),
        "natural": {
            "low": [
                "Spray a strong stream of water on affected leaves to dislodge aphids.",
                "Remove heavily infested leaves manually.",
                "Encourage natural predators such as ladybugs and lacewings.",
            ],
            "medium": [
                "Apply neem oil spray during early morning or evening.",
                "Use insecticidal soap on both upper and lower leaf surfaces.",
                "Control nearby weeds that may host aphids.",
            ],
            "high": [
                "Use repeated neem oil or soap sprays at recommended intervals.",
                "Introduce biological control agents if available.",
                "Isolate severely infested plants to reduce spread.",
            ],
        },
        "chemical": {
            "low": [
                "Chemical control is usually not necessary at low severity.",
            ],
            "medium": [
                "Use a labeled systemic insecticide suitable for aphids as per local agricultural guidance.",
                "Follow crop-specific dosage and pre-harvest interval strictly.",
            ],
            "high": [
                "Use a recommended aphid-specific insecticide under expert supervision.",
                "Rotate mode of action to reduce resistance risk.",
                "Always follow the product label and local agricultural extension advice.",
            ],
        },
        "impact": "Can reduce plant vigor and spread viral diseases if not controlled.",
    },
    "armyworm": {
        "display_name": "Armyworm",
        "description": (
            "Armyworms are leaf-feeding caterpillars that can rapidly damage crop foliage "
            "and spread quickly across fields."
        ),
        "natural": {
            "low": [
                "Hand-pick visible larvae in small-scale fields.",
                "Destroy egg masses and larvae found on leaves.",
                "Use light traps or pheromone traps where suitable.",
            ],
            "medium": [
                "Apply neem-based biopesticides.",
                "Use biological control such as Bacillus thuringiensis (Bt) if suitable for the crop stage.",
                "Keep field borders clean to reduce spread.",
            ],
            "high": [
                "Use repeated biological control measures with close monitoring.",
                "Remove heavily infested plant parts where practical.",
                "Coordinate field-wide control to prevent migration.",
            ],
        },
        "chemical": {
            "low": [
                "Chemical spraying may not be required at low severity if larvae are manually controlled.",
            ],
            "medium": [
                "Use a labeled caterpillar/armyworm control insecticide approved for the crop.",
                "Spray during early larval stages for better effectiveness.",
            ],
            "high": [
                "Apply a recommended insecticide immediately under agricultural guidance.",
                "Ensure proper coverage of infested crop areas.",
                "Avoid repeated use of the same chemistry to reduce resistance.",
            ],
        },
        "impact": "Can cause severe leaf loss and major yield reduction if outbreaks are not controlled early.",
    },
    "grasshopper": {
        "display_name": "Grasshopper",
        "description": (
            "Grasshoppers chew leaves and tender plant parts, causing visible feeding damage "
            "and reduced crop growth."
        ),
        "natural": {
            "low": [
                "Manually remove visible insects in small plots.",
                "Use physical barriers or nets for nursery and garden-scale cultivation.",
                "Maintain field sanitation.",
            ],
            "medium": [
                "Use botanical sprays such as neem-based formulations.",
                "Encourage birds and natural predators where feasible.",
                "Reduce weed growth around the crop.",
            ],
            "high": [
                "Combine botanical measures with field sanitation and trapping.",
                "Use community-level management in large infestations.",
                "Monitor adjoining fields to reduce reinfestation.",
            ],
        },
        "chemical": {
            "low": [
                "Chemical control is often avoidable at low severity.",
            ],
            "medium": [
                "Apply a labeled contact insecticide approved for grasshopper control in the crop.",
                "Use only according to label recommendations.",
            ],
            "high": [
                "Use a recommended insecticide immediately when economic damage is observed.",
                "Follow correct dosage and re-entry interval.",
                "Consult local experts before large-scale spraying.",
            ],
        },
        "impact": "Can cause chewing damage to leaves and reduce photosynthesis.",
    },
    "stemborer": {
        "display_name": "Stemborer",
        "description": (
            "Stemborers attack the internal stem tissues of crops, causing dead hearts, weak growth, "
            "and poor grain formation."
        ),
        "natural": {
            "low": [
                "Remove and destroy infested stems if symptoms are localized.",
                "Use pheromone traps for monitoring adult moth activity.",
                "Maintain proper field sanitation after harvest.",
            ],
            "medium": [
                "Use neem-based products where suitable.",
                "Promote parasitoids and biological control agents if locally available.",
                "Monitor stem damage regularly.",
            ],
            "high": [
                "Destroy heavily infested plants to prevent spread.",
                "Use integrated management combining traps, sanitation, and timely intervention.",
                "Avoid ratooning or leaving infested residues.",
            ],
        },
        "chemical": {
            "low": [
                "Chemical action may not be needed at low infestation levels.",
            ],
            "medium": [
                "Use a labeled systemic insecticide suitable for stemborer control as per local guidance.",
                "Apply at the recommended crop stage for best effectiveness.",
            ],
            "high": [
                "Immediate control is required using approved crop-specific insecticides.",
                "Use only recommended products and doses.",
                "Follow expert recommendations for timing and rotation.",
            ],
        },
        "impact": "Internal feeding causes hidden but serious damage and yield loss.",
    },
    "whitegrub": {
        "display_name": "Whitegrub",
        "description": (
            "Whitegrubs are soil-dwelling larvae that feed on roots, causing wilting, poor growth, "
            "and plant death."
        ),
        "natural": {
            "low": [
                "Collect and destroy visible grubs during soil preparation.",
                "Deep ploughing can expose larvae to predators and sunlight.",
                "Maintain field hygiene and remove host weeds.",
            ],
            "medium": [
                "Use neem cake or other organic soil amendments where recommended.",
                "Encourage birds and natural enemies by exposing soil during field preparation.",
                "Monitor root-zone damage closely.",
            ],
            "high": [
                "Use integrated soil management with organic amendments and repeated field checks.",
                "Treat infested patches separately if possible.",
                "Avoid carrying infested soil to clean areas.",
            ],
        },
        "chemical": {
            "low": [
                "Chemical treatment is usually avoidable at low severity with good field management.",
            ],
            "medium": [
                "Use a labeled soil insecticide only if necessary and crop-appropriate.",
                "Apply as per agricultural extension recommendations.",
            ],
            "high": [
                "Immediate soil-targeted chemical management may be required under expert guidance.",
                "Follow proper dosage, waiting period, and safety measures.",
                "Avoid misuse to reduce environmental harm.",
            ],
        },
        "impact": "Root feeding can lead to stunting, wilting, and serious stand loss.",
    },
    "not_a_pest": {
        "display_name": "Not a Pest",
        "description": "The detected region does not appear to be a target pest class.",
        "natural": [
            "No pest control action is needed for this detection.",
        ],
        "chemical": [
            "No chemical treatment is needed.",
        ],
        "impact": "No harmful pest detected.",
    },
}


# =========================================================
# SEVERITY RULES
# =========================================================
SEVERITY_THRESHOLDS = {
    "aphid": {"low": 3, "medium": 8},
    "armyworm": {"low": 2, "medium": 5},
    "grasshopper": {"low": 2, "medium": 5},
    "stemborer": {"low": 2, "medium": 5},
    "whitegrub": {"low": 2, "medium": 5},
}


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def normalize_pest_name(name: str) -> str:
    return str(name).strip().lower()


def get_severity(pest_name: str, count: int) -> str:
    pest_key = normalize_pest_name(pest_name)

    if pest_key not in SEVERITY_THRESHOLDS:
        return "low"

    low_limit = SEVERITY_THRESHOLDS[pest_key]["low"]
    medium_limit = SEVERITY_THRESHOLDS[pest_key]["medium"]

    if count <= low_limit:
        return "low"
    elif count <= medium_limit:
        return "medium"
    else:
        return "high"


def severity_color(severity: str) -> str:
    if severity == "low":
        return "🟢 Low"
    elif severity == "medium":
        return "🟠 Medium"
    else:
        return "🔴 High"


def get_overall_severity(class_counts: dict) -> str:
    if not class_counts:
        return "low"

    levels = []
    for pest_name, count in class_counts.items():
        if normalize_pest_name(pest_name) == "not_a_pest":
            continue
        levels.append(get_severity(pest_name, count))

    if not levels:
        return "low"
    elif "high" in levels:
        return "high"
    elif "medium" in levels:
        return "medium"
    else:
        return "low"


def generate_farmer_message(class_counts: dict) -> str:
    if not class_counts:
        return (
            "No target pest was detected in the given image. "
            "The crop condition appears safe based on the current prediction."
        )

    filtered_counts = {
        k: v for k, v in class_counts.items() if normalize_pest_name(k) != "not_a_pest"
    }

    if not filtered_counts:
        return (
            "No major target pest was detected. "
            "At present, there is no strong evidence of pest attack in this image."
        )

    major_pest = max(filtered_counts, key=filtered_counts.get)
    major_count = filtered_counts[major_pest]
    overall = get_overall_severity(filtered_counts)

    display_name = PEST_INFO.get(
        normalize_pest_name(major_pest), {}
    ).get("display_name", major_pest.title())

    if overall == "low":
        return (
            f"The image indicates a low level of {display_name} infestation. "
            f"Detected count is about {major_count}. "
            f"At this stage, early monitoring and natural control measures are usually sufficient."
        )
    elif overall == "medium":
        return (
            f"The image indicates a moderate level of {display_name} infestation. "
            f"Detected count is about {major_count}. "
            f"Timely management is recommended to prevent further spread and crop loss."
        )
    else:
        return (
            f"The image indicates a high level of {display_name} infestation. "
            f"Detected count is about {major_count}. "
            f"Immediate action is recommended to reduce serious crop damage."
        )


def get_advisory_for_pest(pest_name: str, count: int):
    pest_key = normalize_pest_name(pest_name)
    info = PEST_INFO.get(pest_key)

    if not info:
        return {
            "severity": "low",
            "natural": ["Monitor the crop regularly."],
            "chemical": ["Consult a local agricultural expert before chemical use."],
            "description": "No detailed pest information available.",
            "impact": "Unknown",
        }

    if pest_key == "not_a_pest":
        return {
            "severity": "low",
            "natural": info["natural"],
            "chemical": info["chemical"],
            "description": info["description"],
            "impact": info["impact"],
        }

    severity = get_severity(pest_key, count)

    return {
        "severity": severity,
        "natural": info["natural"][severity],
        "chemical": info["chemical"][severity],
        "description": info["description"],
        "impact": info["impact"],
    }


def draw_results(image_bgr: np.ndarray, result, class_names: dict) -> np.ndarray:
    drawn = image_bgr.copy()

    if result.boxes is None or len(result.boxes) == 0:
        return drawn

    boxes = result.boxes.xyxy.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)

    for box, conf, cls_id in zip(boxes, confs, classes):
        x1, y1, x2, y2 = box
        class_label = class_names[int(cls_id)]
        label = f"{class_label} {conf:.2f}"

        cv2.rectangle(drawn, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            drawn,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 180, 0),
            2,
        )

    return drawn


def run_prediction(pil_image: Image.Image, model):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        temp_path = tmp.name
        pil_image.save(temp_path)

    result = model.predict(source=temp_path, conf=0.25, verbose=False)[0]
    image_bgr = cv2.imread(temp_path)

    if os.path.exists(temp_path):
        os.remove(temp_path)

    class_names = model.names
    plotted_bgr = draw_results(image_bgr, result, class_names)

    class_counts = {}
    total_count = 0

    if result.boxes is not None and len(result.boxes) > 0:
        pred_classes = result.boxes.cls.cpu().numpy().astype(int)
        counts = Counter(pred_classes)
        total_count = len(pred_classes)
        class_counts = {class_names[k]: v for k, v in counts.items()}

    return plotted_bgr, total_count, class_counts


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("⚙️ Settings")

model_path = st.sidebar.text_input("Model Path", value=DEFAULT_MODEL_PATH)

if not os.path.exists(model_path):
    st.sidebar.warning("Model file not found. Please update the path to your YOLOv8 model.")
    st.stop()

model = load_model(model_path)
st.sidebar.success("YOLOv8 model loaded successfully.")

input_mode = st.sidebar.radio("Select Input Mode", ["Upload Image", "Use Camera"])

st.sidebar.markdown("---")
st.sidebar.markdown("### Supported Pest Classes")
for pest in model.names.values():
    st.sidebar.write(f"- {pest}")


# =========================================================
# INPUT SECTION
# =========================================================
image = None

if input_mode == "Upload Image":
    uploaded_file = st.file_uploader(
        "Upload a pest image",
        type=["jpg", "jpeg", "png", "webp"],
    )
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

elif input_mode == "Use Camera":
    captured_file = st.camera_input("Capture image using camera")
    if captured_file is not None:
        image = Image.open(captured_file).convert("RGB")


# =========================================================
# MAIN INFERENCE
# =========================================================
if image is not None:
    st.subheader("📷 Input Image")
    st.image(image, caption="Input Image", use_container_width=True)

    with st.spinner("Detecting pests and generating advisory..."):
        plotted_bgr, total_count, class_counts = run_prediction(image, model)
        plotted_rgb = bgr_to_rgb(plotted_bgr)

    st.subheader("✅ Predicted Output")
    st.image(
        plotted_rgb,
        caption="Detected Pests with Bounding Boxes",
        use_container_width=True,
    )

    st.subheader("📊 Detection Summary")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total Detected Pests", total_count)

    with col2:
        st.metric("Overall Severity", severity_color(get_overall_severity(class_counts)))

    if class_counts:
        summary_data = []
        for pest_name, count in class_counts.items():
            summary_data.append(
                {
                    "Pest Class": pest_name,
                    "Count": count,
                    "Severity": severity_color(get_severity(pest_name, count)),
                }
            )

        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    else:
        st.info("No target pest detected in this image.")

    st.subheader("🧑‍🌾 Farmer Advisory Summary")
    st.success(generate_farmer_message(class_counts))

    if class_counts:
        st.subheader("🌿 Pest-wise Recommendations")

        for pest_name, count in class_counts.items():
            advisory = get_advisory_for_pest(pest_name, count)
            display_name = PEST_INFO.get(
                normalize_pest_name(pest_name), {}
            ).get("display_name", pest_name.title())

            with st.expander(
                f"{display_name} — Count: {count} — Severity: {severity_color(advisory['severity'])}",
                expanded=True,
            ):
                st.markdown(f"**About this pest:** {advisory['description']}")
                st.markdown(f"**Impact on crop:** {advisory['impact']}")

                st.markdown("**Natural Control Suggestions:**")
                for item in advisory["natural"]:
                    st.write(f"- {item}")

                st.markdown("**Chemical Control Suggestions:**")
                for item in advisory["chemical"]:
                    st.write(f"- {item}")

    st.markdown("---")
    st.warning(
        "⚠️ Chemical suggestions in this app are general advisory only. "
        "Before applying any pesticide, always verify crop suitability, local regulations, "
        "dosage, waiting period, and safety instructions with a qualified agricultural expert."
    )

else:
    st.info("Upload or capture an image to start pest detection.")
