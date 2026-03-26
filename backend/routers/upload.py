import uuid
import io
import os
import json
import pandas as pd
import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Depends
from fastapi.responses import JSONResponse
from models.db_models import User
from routers.auth import get_current_user

from models.schemas import UploadResponse, ProtectedAttribute
from core.dataset_generator import generate_biased_dataset, get_builtin_dataset_info

router = APIRouter()

# In-memory dataset store (use Redis/DB in production)
_dataset_store: dict = {}


def _load_file(filename: str, contents: bytes) -> pd.DataFrame:
    """
    Auto-detect file format and load into DataFrame.
    Supports: CSV, TSV, TXT, Excel, JSON, Parquet, Feather.
    Falls back latin-1 encoding on UTF-8 decode errors (CLI FIX).
    """
    ext = os.path.splitext(filename)[1].lower()

    if ext in (".csv", ".txt"):
        try:
            df = pd.read_csv(io.BytesIO(contents), encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(contents), encoding="latin-1")

    elif ext == ".tsv":
        try:
            df = pd.read_csv(io.BytesIO(contents), sep="\t", encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(contents), sep="\t", encoding="latin-1")

    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(io.BytesIO(contents))

    elif ext == ".json":
        try:
            df = pd.read_json(io.BytesIO(contents))
        except ValueError:
            data = json.loads(contents)
            df = pd.json_normalize(data)

    elif ext == ".parquet":
        df = pd.read_parquet(io.BytesIO(contents))

    elif ext == ".feather":
        df = pd.read_feather(io.BytesIO(contents))

    else:
        raise ValueError(
            f"Unsupported format: '{ext}'. "
            "Supported: .csv .tsv .txt .xlsx .xls .json .parquet .feather"
        )

    # Auto-coerce object columns that should be numeric (CLI FIX-7)
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = pd.to_numeric(df[col], errors="raise")
            except Exception:
                pass

    return df


PROTECTED_KEYWORDS = {
    "gender": 0.98, "sex": 0.96, "race": 0.97, "ethnicity": 0.95,
    "age": 0.88, "zip": 0.72, "zipcode": 0.75, "zip_code": 0.78,
    "religion": 0.91, "nationality": 0.85, "disability": 0.88,
    "marital": 0.70, "pregnant": 0.85,
}


def _detect_protected_attrs(columns: list) -> list:
    results = []
    for col in columns:
        col_lower = col.lower().replace("_", "").replace("-", "")
        for keyword, confidence in PROTECTED_KEYWORDS.items():
            kw_clean = keyword.replace("_", "")
            if kw_clean in col_lower:
                results.append(ProtectedAttribute(
                    column=col,
                    confidence=round(confidence + np.random.uniform(-0.02, 0.02), 3),
                    detected_type=keyword,
                ))
                break
    return results


@router.post("/upload", response_model=UploadResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    """Upload a CSV, TSV, Excel, JSON, Parquet, or Feather dataset for bias analysis."""
    dataset_id = str(uuid.uuid4())

    try:
        contents = await file.read()
        filename = file.filename or "dataset.csv"

        df = _load_file(filename, contents)

        if len(df) < 20:
            raise HTTPException(status_code=400, detail="Dataset too small. Minimum 20 rows required.")

        # Store dataset
        _dataset_store[dataset_id] = df

        # Detect protected attributes
        detected = _detect_protected_attrs(list(df.columns))

        # Preview rows
        preview = df.head(10).fillna("").to_dict(orient="records")

        return UploadResponse(
            dataset_id=dataset_id,
            filename=filename,
            rows=len(df),
            columns=list(df.columns),
            detected_protected_attrs=detected,
            preview_rows=preview,
        )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@router.get("/builtin-datasets")
async def get_builtin_datasets():
    """Return info about built-in sample datasets."""
    return get_builtin_dataset_info()


@router.post("/load-builtin")
async def load_builtin_dataset(dataset_id_name: str = Form(...)):
    """Load a built-in dataset (compas/adult/german) and return dataset_id."""
    dataset_id = str(uuid.uuid4())

    try:
        if dataset_id_name == "compas":
            # ProPublica COMPAS two-year recidivism dataset
            _URL = (
                "https://raw.githubusercontent.com/propublica/"
                "compas-analysis/master/compas-scores-two-years.csv"
            )
            raw = pd.read_csv(_URL)
            _keep = [
                "sex", "age", "race", "juv_fel_count", "juv_misd_count",
                "juv_other_count", "priors_count", "c_charge_degree", "two_year_recid",
            ]
            df = raw[[c for c in _keep if c in raw.columns]].dropna()

        elif dataset_id_name == "adult":
            # UCI Adult Income dataset (predict >50K salary)
            _ADULT_COLS = [
                "age", "workclass", "fnlwgt", "education", "education_num",
                "marital_status", "occupation", "relationship", "race", "sex",
                "capital_gain", "capital_loss", "hours_per_week", "native_country", "income",
            ]
            df = pd.read_csv(
                "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
                names=_ADULT_COLS, na_values=" ?", skipinitialspace=True,
            ).dropna()
            df["income"] = (df["income"].str.strip() == ">50K").astype(int)

        elif dataset_id_name == "german":
            # UCI German Credit dataset (1 = good credit, 2 = bad credit)
            _GER_COLS = [f"attr_{i}" for i in range(1, 21)] + ["credit_risk"]
            df = pd.read_csv(
                "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data",
                sep=" ", names=_GER_COLS, header=None,
            )
            df["credit_risk"] = (df["credit_risk"] == 1).astype(int)
        elif dataset_id_name == "synthetic_hiring":
            df = generate_biased_dataset(1000, bias_level=0.65, domain="hiring")
        elif dataset_id_name == "synthetic_lending":
            df = generate_biased_dataset(1000, bias_level=0.60, domain="lending")
        elif dataset_id_name == "synthetic_medical":
            df = generate_biased_dataset(1000, bias_level=0.55, domain="medical")
        else:
            raise HTTPException(status_code=404, detail=f"Unknown built-in dataset: {dataset_id_name}")

        _dataset_store[dataset_id] = df
        detected = _detect_protected_attrs(list(df.columns))
        preview = df.head(10).fillna("").to_dict(orient="records")

        return UploadResponse(
            dataset_id=dataset_id,
            filename=f"{dataset_id_name}.csv",
            rows=len(df),
            columns=list(df.columns),
            detected_protected_attrs=detected,
            preview_rows=preview,
        )

    except HTTPException:
        raise
    except Exception as e:
        # Fallback to synthetic if AIF360 dataset fails (e.g., download issue)
        domain_map = {"compas": "criminal_justice", "adult": "hiring", "german": "lending"}
        domain = domain_map.get(dataset_id_name, "hiring")
        df = generate_biased_dataset(1000, bias_level=0.65, domain=domain if domain != "criminal_justice" else "hiring")
        _dataset_store[dataset_id] = df
        detected = _detect_protected_attrs(list(df.columns))
        preview = df.head(10).fillna("").to_dict(orient="records")
        return UploadResponse(
            dataset_id=dataset_id,
            filename=f"{dataset_id_name}_synthetic.csv",
            rows=len(df),
            columns=list(df.columns),
            detected_protected_attrs=detected,
            preview_rows=preview,
        )


def get_dataset(dataset_id: str) -> pd.DataFrame:
    """Helper to retrieve dataset from store."""
    if dataset_id not in _dataset_store:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found. Please upload again.")
    return _dataset_store[dataset_id]
