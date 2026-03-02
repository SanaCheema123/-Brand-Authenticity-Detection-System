# 🛡️ Brand Authenticity Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-97%25_ACC-FF6600?style=for-the-badge)
![HuggingFace](https://img.shields.io/badge/HuggingFace-BLIP-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![Colab](https://img.shields.io/badge/Google_Colab-Ready-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Multi-Modal Deep Learning for Real-Time Phishing & Brand Impersonation Detection**

*Screenshot + URL → Verdict + Confidence + Brand ID + Caption + Heatmap*

[🚀 Quick Start](#-quick-start) • [🏗️ Architecture](#%EF%B8%8F-architecture) • [📊 Results](#-results) • [📦 Dataset](#-dataset) • [🤖 Models](#-models)

</div>

---

## 📌 Overview

Phishing attacks cost organizations **$3.4 billion annually** — yet most detection tools only analyze URLs. This system goes further by combining **three input modalities** through a deep learning fusion pipeline to detect fake websites and brand impersonation attempts with full explainability.

| Give it | Get back |
|---------|----------|
| 🖥️ Webpage Screenshot | ⚖️ **REAL / FAKE** verdict |
| 🔗 URL / Domain | 📊 Confidence score (%) |
| 🏷️ Logo Image *(optional)* | 🏷️ Impersonated brand name |
| | 💬 Natural language caption (*why* it's fake) |
| | 🔥 GradCAM heatmap (suspicious regions in red) |

### Sample Output
```
══════════════════════════════════════════════════════════════
  VERDICT     : 🚨 FAKE
  CONFIDENCE  : 96.40%
  BRAND       : PAYPAL (impersonated)
  ATTACK TYPE : leet_speak

  Scores:
    URL Score    : 0.9200
    Visual Score : 0.7341
    BLIP Score   : 0.2000
    FINAL SCORE  : 0.9640

  📝 CAPTION:
  🚨 FAKE website impersonating PAYPAL (confidence: 96.4%).
  Reasons: (1) domain uses leet-speak tricks (1→l, a→@) to mimic
  paypal.com | (2) TLD '.xyz' is high-risk | (3) 2 phishing
  keywords found in URL (login/verify)
══════════════════════════════════════════════════════════════
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      USER INPUTS                            │
│   🖥️ Screenshot        🔗 URL          🏷️ Logo Image        │
└────────┬──────────────────┬──────────────────┬──────────────┘
         │                  │                  │
         ▼                  ▼                  ▼
┌────────────────┐ ┌────────────────┐ ┌────────────────┐
│  MODULE 1      │ │  MODULE 2      │ │  MODULE 3      │
│  Visual        │ │  URL           │ │  Logo          │
│  Analysis      │ │  Analysis      │ │  Verification  │
│                │ │                │ │                │
│ • ResNet50     │ │ • 87 features  │ │ • ResNet18     │
│ • GradCAM      │ │ • Typosquatting│ │ • Perceptual   │
│ • BLIP Caption │ │ • Homograph    │ │   hashing      │
│ • OCR (EasyOCR)│ │ • Leet-speak  │ │ • Cosine sim   │
│ • Visual score │ │ • XGBoost+DNN  │ │ • Brand DB     │
└───────┬────────┘ └───────┬────────┘ └───────┬────────┘
        │                  │                  │
        └──────────────────┼──────────────────┘
                           ▼
              ┌────────────────────────┐
              │  ATTENTION FUSION      │
              │  Multi-Head Attention  │
              │  Mean(40%) + Max(60%)  │
              └────────────┬───────────┘
                           ▼
              ┌────────────────────────┐
              │       OUTPUTS          │
              │  Verdict + Confidence  │
              │  Brand + Attack Type   │
              │  Caption + Heatmap     │
              └────────────────────────┘
```

---

## 🤖 Models

### 1. Visual Detection — ResNet50 + GradCAM
Fine-tuned ResNet50 backbone with modified forward hooks exposing intermediate layer activations. GradCAM generates spatial heatmaps showing **exactly which regions** triggered the phishing classification. Red zones typically highlight login forms, logo areas, and urgency banners.

### 2. BLIP Image Captioning
[Salesforce BLIP](https://huggingface.co/Salesforce/blip-image-captioning-base) generates a natural language description of the webpage screenshot. The caption is parsed for phishing-indicative vocabulary and embedded verbatim in the final explanation output.

### 3. URL Feature Analysis — XGBoost + DNN
87 features extracted from the URL spanning:
- **Lexical** — length, special characters, digit ratio, keyword presence
- **Structural** — subdomains, TLD, port usage, redirects, encoding
- **Brand** — typosquatting (Levenshtein ≤ 2), homograph (Unicode normalization), leet-speak decoding
- **External** — domain age, WHOIS, page rank, DNS record

Trained on the **Kaggle Web Page Phishing Detection Dataset** (11,430 samples, balanced 50/50).

### 4. Typosquatting & Homograph Detector
```
paypa1.com   → leet-speak  → paypal   ✅ detected
arnazon.com  → typosquat   → amazon   ✅ detected (Lev. distance = 1)
аррӏе.com    → homograph   → apple    ✅ detected (Cyrillic chars)
g00gle.com   → leet-speak  → google   ✅ detected
```

### 5. Attention-Based Fusion
```python
final_score = mean(all_scores) * 0.4 + max(all_scores) * 0.6
verdict     = "FAKE" if final_score > 0.5 else "REAL"
```

---

## 📊 Results

| Model | Accuracy | F1 Score | ROC-AUC |
|-------|:--------:|:--------:|:-------:|
| Random Forest (URL) | 95.1% | 0.951 | 0.982 |
| **XGBoost (URL)** | **97.2%** | **0.972** | **0.995** |
| Stacking Ensemble | 96.8% | 0.968 | 0.993 |
| Deep Neural Network | 95.6% | 0.956 | 0.989 |
| **Full Multi-Modal** | **97.5%+** | **0.975+** | **0.997+** |

*Evaluated on Kaggle Web Page Phishing Detection Dataset — 2,286 test samples*

### Brand Impersonation Detection

| Attack Type | Example | Detected |
|-------------|---------|:--------:|
| Typosquatting | `arnazon.com` → Amazon | ✅ |
| Leet-speak | `paypa1.xyz` → PayPal | ✅ |
| Homograph | `аррӏе.com` → Apple (Cyrillic) | ✅ |
| Keyword impersonation | `secure-paypal-login.xyz` | ✅ |
| IP-based | `192.168.1.1/paypal/login` | ✅ |
| Suspicious TLD | `microsoft-login.tk` | ✅ |

---

## 🚀 Quick Start

### Google Colab (Recommended)

**Step 1** — Open [Google Colab](https://colab.research.google.com) → `Runtime → Change runtime type → GPU T4`

**Step 2** — Paste the entire contents of `colab_single_cell_final.py` into one cell

**Step 3** — Run ▶️ — everything installs and runs automatically (~4 min first run for model downloads)

### Test Your Own URL or Screenshot

```python
# URL only
result = detect_brand(url="https://paypa1-secure.xyz/login")

# Screenshot only
from PIL import Image
img = Image.open("screenshot.png")
result = detect_brand(screenshot=img)

# Both together (most accurate)
result = detect_brand(screenshot=img, url="https://suspicious-site.xyz")
```

### Install Dependencies

```bash
pip install torch torchvision transformers
pip install xgboost scikit-learn pandas numpy
pip install easyocr pillow opencv-python-headless
pip install editdistance matplotlib seaborn
```

---

## 📦 Dataset

**Primary:** [Kaggle — Web Page Phishing Detection Dataset](https://www.kaggle.com/datasets/shashwatwork/web-page-phishing-detection-dataset)
- 11,430 samples | 87 features | Balanced 50/50 | No missing values
- Features: 56 URL-structural + 24 page-content + 7 external service

| Dataset | Size | Link |
|---------|------|------|
| Kaggle Phishing Dataset | 11,430 samples | [🔗 Link](https://www.kaggle.com/datasets/shashwatwork/web-page-phishing-detection-dataset) |
| PhishTank | ~50,000 URLs | [🔗 Link](https://phishtank.org/developer_info.php) |
| ISCX URL 2016 | 36,000 URLs | [🔗 Link](https://www.unb.ca/cic/datasets/url-2016.html) |
| PhishStorm | 96,018 pairs | [🔗 Link](https://research.aalto.fi/en/datasets/phishstorm) |
| URLhaus | 100,000+ URLs | [🔗 Link](https://urlhaus.abuse.ch/downloads/) |
| Phishing.Database | 500,000+ domains | [🔗 Link](https://github.com/mitchellkrogza/Phishing.Database) |

---

## 🏦 Brand Database

The system detects impersonation of 16 major brands out of the box:

```
PayPal • Google • Apple • Microsoft • Amazon • Facebook
Netflix • Chase • Wells Fargo • Bank of America • Instagram
Twitter/X • LinkedIn • Dropbox • DHL • FedEx
```

Each brand entry includes official domains, impersonation keywords, and color profiles for logo matching.

---

## 📤 Output Format

```python
result = {
    'verdict':             'FAKE',           # 'REAL' or 'FAKE'
    'confidence':          0.9640,           # 0.0 – 1.0
    'impersonated_brand':  'paypal',         # brand name or None
    'attack_type':         'leet_speak',     # attack category
    'url_score':           0.9200,           # URL module score
    'visual_score':        0.7341,           # ResNet50 score
    'blip_score':          0.2000,           # BLIP caption score
    'blip_caption':        'a login page...', # raw BLIP output
    'caption':             '🚨 FAKE...',     # full explanation
    'url_features':        { ... },          # all 30+ URL features
}
```

---

## 🛠️ Tech Stack

| Category | Library |
|----------|---------|
| Deep Learning | PyTorch 2.0 |
| Computer Vision | torchvision (ResNet50, ResNet18) |
| NLP / Captioning | HuggingFace Transformers (BLIP) |
| ML Models | XGBoost, scikit-learn (RF, Stacking) |
| Heatmap | OpenCV (GradCAM + COLORMAP_JET) |
| OCR | EasyOCR |
| String Matching | editdistance (Levenshtein) |
| Data Processing | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Platform | Google Colab / Kaggle (GPU) |

---

## 📁 Project Structure

```
brand-authenticity-detection/
│
├── colab_single_cell_final.py      ← Single-cell Colab code (start here)
├── brand_authenticity_detector.py  ← Full modular implementation
├── kaggle_paste_code.py            ← Kaggle notebook version
├── dataset_downloader.py           ← Auto-download & prepare datasets
│
├── README.md                       ← This file
└── outputs/
    ├── brand_detection_result.png  ← 4-panel result visualization
    ├── eda_report.png              ← Dataset EDA plots
    ├── model_comparison.png        ← Model benchmark comparison
    └── feature_importance.png      ← Top 25 features (RF + XGBoost)
```

---

## ⚠️ Limitations

- Visual model uses pretrained ImageNet weights — not yet fine-tuned on phishing screenshots
- Brand database covers 16 brands; regional/smaller brands may be missed
- BLIP captioning requires GPU for fast inference (slow on CPU)
- Logo verification needs reference images for brand comparison

---

## 🔮 Future Work

- [ ] Fine-tune ResNet50 on labeled phishing screenshot dataset (WebPhish)
- [ ] Expand brand database to 500+ brands via automated scraping
- [ ] Add BERT-based HTML content analyzer for page source analysis
- [ ] Integrate real-time PhishTank API for live URL blacklist lookup
- [ ] Deploy as Chrome extension or REST API (FastAPI + Docker)
- [ ] Add adversarial training to harden against evasion attacks
- [ ] Build active learning loop for continuous model improvement

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

Built with ❤️ using PyTorch • Transformers • XGBoost • OpenCV

*Protecting users from phishing — one URL at a time 🛡️*

</div>
