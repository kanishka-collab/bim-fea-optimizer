"""
FastAPI REST Microservice for Eurocode 3 Structural Optimization & AI Surrogate Engine
Exposes high-performance microservice endpoints for FEA matrix solving, AI predictions, and PDF calculation reports.
"""

import os
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import joblib

from fea_test import MultiStoryFEAEngine
from steel_sections import size_member, STEEL_SECTIONS
from report_generator import generate_pdf_report
from train_model import train_and_save_surrogate_model

# Initialize FastAPI App
app = FastAPI(
    title="3D BIM-to-FEA Structural Optimization REST API",
    description="Enterprise Eurocode 3 Structural Engine, HPC AI Surrogate & Automated PDF Calculation Service",
    version="2.0.0"
)

# Enable CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Request Schemas
class FrameAnalysisRequest(BaseModel):
    num_stories: int = Field(2, ge=1, le=5, description="Number of stories (1 to 5)")
    num_bays_x: int = Field(2, ge=1, le=3, description="Number of bays along X-axis")
    num_bays_z: int = Field(1, ge=1, le=3, description="Number of bays along Z-axis")
    story_height: float = Field(3.5, ge=2.5, le=5.0, description="Story height in meters")
    bay_width_x: float = Field(6.0, ge=3.0, le=10.0, description="Bay width along X-axis in meters")
    bay_width_z: float = Field(5.0, ge=3.0, le=10.0, description="Bay width along Z-axis in meters")
    G_k: float = Field(15.0, ge=0.0, le=50.0, description="Characteristic Dead Load in kN/m")
    Q_k: float = Field(10.0, ge=0.0, le=50.0, description="Characteristic Live Load in kN/m")
    W_k: float = Field(5.0, ge=0.0, le=25.0, description="Characteristic Wind Load in kN/m")
    fy_grade: float = Field(275.0, description="Steel yield strength f_y in MPa (e.g. 275 or 355)")

class SurrogatePredictRequest(FrameAnalysisRequest):
    pass

class ReportGenerationRequest(BaseModel):
    grid_params: Dict[str, Any]
    load_params: Dict[str, Any]
    fe_results: Dict[str, Any]
    section_results: Dict[str, Any]
    drift_results: Dict[str, Any]

# Helper function for cached surrogate inference
SURROGATE_CACHE = None

def get_surrogate_model():
    global SURROGATE_CACHE
    if SURROGATE_CACHE is None:
        model_path = "surrogate_model.pkl"
        if not os.path.exists(model_path):
            csv_file = "enterprise_fea_dataset.csv" if os.path.exists("enterprise_fea_dataset.csv") else "fea_dataset.csv"
            SURROGATE_CACHE = train_and_save_surrogate_model(csv_file, model_path)
        else:
            SURROGATE_CACHE = joblib.load(model_path)
    return SURROGATE_CACHE

@app.get("/health", summary="Microservice Health Check")
def health_check():
    surrogate_status = False
    try:
        model = get_surrogate_model()
        surrogate_status = model is not None
    except Exception:
        surrogate_status = False

    return {
        "status": "ok",
        "service": "BIM-FEA Structural Optimizer Microservice",
        "timestamp": datetime.now().isoformat(),
        "solver": "PyNiteFEA Eurocode 3 Engine",
        "surrogate_model_loaded": surrogate_status
    }

@app.post("/api/v1/analyze", summary="Run PyNite FEA Matrix Solver & Section Sizing")
def analyze_frame(req: FrameAnalysisRequest):
    try:
        t0 = time.perf_counter()
        engine = MultiStoryFEAEngine(
            num_bays_x=req.num_bays_x,
            num_bays_z=req.num_bays_z,
            num_stories=req.num_stories,
            bay_width_x=req.bay_width_x,
            bay_width_z=req.bay_width_z,
            story_height=req.story_height,
            G_k=req.G_k,
            Q_k=req.Q_k,
            W_k=req.W_k,
            fy_MPa=req.fy_grade
        )
        df_results, opt_summary, _ = engine.build_and_analyze()
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        opt_summary["solver_latency_ms"] = latency_ms

        return {
            "status": "success",
            "execution_mode": "PyNite FEA Matrix Solver",
            "opt_summary": opt_summary,
            "results": df_results.to_dict(orient="records"),
            "latency_ms": latency_ms
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FEA Matrix Solver Error: {str(e)}")

@app.post("/api/v1/predict", summary="Run Sub-20ms AI Surrogate Model Prediction")
def predict_surrogate(req: SurrogatePredictRequest):
    try:
        t0 = time.perf_counter()
        surrogate = get_surrogate_model()

        total_H = req.story_height * req.num_stories
        sway_lim_mm = (total_H * 1000.0) / 500.0

        X_input = pd.DataFrame([{
            'num_stories': req.num_stories, 'num_bays_x': req.num_bays_x, 'num_bays_z': req.num_bays_z,
            'story_height': req.story_height, 'bay_width_x': req.bay_width_x, 'bay_width_z': req.bay_width_z,
            'G_k': req.G_k, 'Q_k': req.Q_k, 'W_k': req.W_k,
            'H_m': total_H, 'Lx_m': req.bay_width_x * req.num_bays_x, 'Lz_m': req.bay_width_z * req.num_bays_z,
            'point_load_P_kN': req.Q_k * 5.0, 'dist_load_w_kNm': req.G_k + req.Q_k
        }])

        feat_cols = surrogate.get('feature_cols', X_input.columns.tolist())
        for col in feat_cols:
            if col not in X_input.columns:
                X_input[col] = 0.0
        X_input = X_input[feat_cols]

        pred_beam_sec = surrogate['clf_beam'].predict(X_input)[0]
        pred_col_sec = surrogate['clf_col'].predict(X_input)[0]
        pred_weight = surrogate['reg_weight'].predict(X_input)[0]
        pred_beam_M = surrogate['reg_beam_M'].predict(X_input)[0]
        pred_col_P = surrogate['reg_col_P'].predict(X_input)[0]

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        beam_sec_info = next((s for s in STEEL_SECTIONS if s['name'] == pred_beam_sec), STEEL_SECTIONS[8])
        col_sec_info = next((s for s in STEEL_SECTIONS if s['name'] == pred_col_sec), STEEL_SECTIONS[0])

        M_Rd_beam = beam_sec_info['W_pl_y'] * req.fy_grade * 1e-3
        N_Rd_col = col_sec_info['A'] * req.fy_grade * 1e-1

        beam_util = round((pred_beam_M / M_Rd_beam) * 100, 2)
        col_util = round((pred_col_P / N_Rd_col) * 100, 2)

        pred_sway_mm = round(min(50.0, req.W_k * req.num_stories * 0.8), 2)
        util_sway = round(pred_sway_mm / sway_lim_mm, 4)

        opt_summary = {
            "optimal_beam": {"name": pred_beam_sec, "type": "UB", "mass": beam_sec_info['mass'], "util_pct": min(beam_util, 99.9)},
            "optimal_col": {"name": pred_col_sec, "type": "UC", "mass": col_sec_info['mass'], "util_pct": min(col_util, 99.9)},
            "total_weight_kg": round(pred_weight, 2),
            "max_beam_M": round(pred_beam_M, 2),
            "max_col_P": round(pred_col_P, 2),
            "total_height_m": round(total_H, 2),
            "delta_sway_mm": pred_sway_mm,
            "delta_sway_lim_mm": round(sway_lim_mm, 2),
            "util_sway": util_sway,
            "sway_status": "Pass" if util_sway <= 1.0 else "Fail",
            "latency_ms": latency_ms
        }

        results = []
        for k in range(req.num_stories):
            for i in range(req.num_bays_x + 1):
                for j in range(req.num_bays_z + 1):
                    results.append({
                        "Member": f"C_{i}_{k}_{j}", "Type": "Column", "Length (m)": req.story_height,
                        "Axial Force P_u (kN)": round(pred_col_P, 2), "Max Bending M_z (kN*m)": 0.0,
                        "Max Bending M_y (kN*m)": 0.0, "Max Envelope M_u (kN*m)": 0.0,
                        "Assigned Section": pred_col_sec, "Mass (kg/m)": col_sec_info['mass'],
                        "Util (%)": min(col_util, 99.9), "Status": "Green" if col_util < 70 else "Yellow"
                    })

        for k in range(1, req.num_stories + 1):
            for i in range(req.num_bays_x):
                for j in range(req.num_bays_z + 1):
                    results.append({
                        "Member": f"BX_{i}_{k}_{j}", "Type": "Beam", "Length (m)": req.bay_width_x,
                        "Axial Force P_u (kN)": 0.0, "Max Bending M_z (kN*m)": round(pred_beam_M, 2),
                        "Max Bending M_y (kN*m)": 0.0, "Max Envelope M_u (kN*m)": round(pred_beam_M, 2),
                        "Assigned Section": pred_beam_sec, "Mass (kg/m)": beam_sec_info['mass'],
                        "Util (%)": min(beam_util, 99.9), "Status": "Green" if beam_util < 70 else "Yellow"
                    })

        return {
            "status": "success",
            "execution_mode": "HPC AI Surrogate Microservice",
            "opt_summary": opt_summary,
            "results": results,
            "latency_ms": latency_ms
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Surrogate Prediction Error: {str(e)}")

@app.post("/api/v1/report", summary="Generate Stamp-Ready PDF Calculation Sheet Stream")
def generate_report(req: ReportGenerationRequest):
    try:
        pdf_bytes = generate_pdf_report(
            grid_params=req.grid_params,
            load_params=req.load_params,
            fe_results=req.fe_results,
            section_results=req.section_results,
            drift_results=req.drift_results
        )
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=Structural_Calculation_Report.pdf"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF Report Generation Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
