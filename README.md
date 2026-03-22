# FairSight: AI Bias Detection & Fairness Platform ⚖️

FairSight is a production-grade web application for detecting, analyzing, and mitigating algorithmic bias in machine learning models and datasets.

It provides a comprehensive suite of fairness metrics, proxy detection, intersectional analysis, actionable LLM narratives via Gemini, and automated compliance reporting.

## 🚀 Features

*   **7-Metric Fairness Audit:** Calculates Disparate Impact, Statistical Parity, Equal Opportunity, Average Odds, Theil Index, Individual Fairness, and Calibration.
*   **Proxy Variable Detection:** Identifies hidden proxies using Cramér's V and Pearson correlations.
*   **Intersectional Bias Analysis:** Finds "invisible" bias that occurs only at the intersection of multiple protected groups (e.g., Black Women).
*   **Bias Mitigation Engine:** Apply Reweighing, Disparate Impact Remover, Threshold Adjustment, and Adversarial Debiasing to repair datasets/models.
*   **Story Mode:** Generates human-centric counterfactual stories showing how the algorithm behaves differently based *only* on a protected attribute.
*   **Gemini AI Narrator:** Translates complex math into plain-English, multi-lingual executive summaries and action plans.
*   **Compliance PDF Reports:** Generates board-ready PDF certificates mapped to EEOC, EU AI Act, and ECOA guidelines.
*   **Built-in & Synthetic Datasets:** Start immediately without your own data using synthetic Hiring, Lending, and Medical datasets or classic ML fairness datasets (COMPAS, Adult, German).

## 🛠️ Tech Stack

*   **Backend:** Python, FastAPI, Pandas, Scikit-Learn, AIF360, SHAP, ReportLab, Google Generative AI (Gemini).
*   **Frontend:** React (React Router v6), Tailwind-inspired custom CSS, Chart.js, Framer Motion, Axios.
*   **Deployment:** Docker, Docker Compose.

## 🏃‍♂️ Getting Started

### Prerequisites

*   Python 3.10+
*   Node.js 18+
*   A Gemini API Key (from Google AI Studio)

### 1. Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
pip install -r requirements.txt
```

Create a `.env` file in the `backend` directory and add your Gemini API key:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

Start the backend server:

```bash
uvicorn main:app --reload --port 8000
```

### 2. Frontend Setup

In a new terminal:

```bash
cd frontend
npm install
npm start
```

The application will be available at `http://localhost:3000`.

### 3. Using Docker Compose (Alternative)

Ensure Docker Desktop is installed.
Ensure `.env` exists in `backend/` with `GEMINI_API_KEY`.

```bash
docker-compose up --build
```

Access the frontend at `http://localhost:3000` and backend API documentation at `http://localhost:8000/docs`.

## 📖 Usage Flow

1.  **Upload:** Drop a CSV/JSON file or select a built-in dataset (e.g., Synthetic Hiring).
2.  **Configure:** Confirm auto-detected protected attributes and select the outcome column.
3.  **Analyze (Dashboard):** View the Bias Fingerprint, proxy heatmap, intersectional table, and read the Gemini-generated narrative.
4.  **Story Mode:** See counterfactual profiles demonstrating the bias on individuals.
5.  **Mitigate:** Apply fairness algorithms (e.g., Reweighing) and view the before/after results and the fairness vs. accuracy trade-off curve.
6.  **Simulate / Monitor:** Play with live sliders to simulate demographic shifts or view simulated production drift over time.
7.  **Report:** Download a PDF compliance certificate summarizing the audit.

## 📄 License

MIT License
