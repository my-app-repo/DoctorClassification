import pickle
from pathlib import Path
from typing import Dict, Any
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_from_directory

APP_DIR = Path(__file__).parent
MODELS_DIR = APP_DIR / "deploy"

def load_metadata() -> Dict[str, Any]:
    metadata_path = MODELS_DIR / "model_metadata.pkl"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Model metadata not found at {metadata_path}.")
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    return metadata

def list_available_models() -> Dict[str, str]:
    if not MODELS_DIR.exists():
        return {}
    models = {}
    for file in MODELS_DIR.glob("*_classifier.pkl"):
        name = file.stem.replace("_classifier", "")
        models[name] = str(file)
    return models

def load_model(model_name: str):
    model_path = MODELS_DIR / f"{model_name}_classifier.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model '{model_name}' not found at {model_path}.")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

def coerce_features_to_dataframe(payload_features: Dict[str, Any], feature_names):
    # Accept either dict (single row) or list of dicts (multiple rows)
    if isinstance(payload_features, dict):
        rows = [payload_features]
    elif isinstance(payload_features, list):
        rows = payload_features
    else:
        raise ValueError("'features' must be an object or an array of objects")

    df = pd.DataFrame(rows)
    
    # Check if all values are empty/NaN
    all_empty = True
    for col in df.columns:
        if col in df.columns and df[col].iloc[0] != '' and pd.notna(df[col].iloc[0]):
            all_empty = False
            break
    
    if all_empty:
        raise ValueError("All feature values are empty. Please provide at least one valid feature value.")
    
    # Add any missing columns with 0, reorder to training order
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    
    df = df[feature_names]
    
    # Coerce numeric types where possible and handle empty strings
    for c in df.columns:
        if df[c].dtype == object:
            # Replace empty strings with 0 and handle conversion properly
            df[c] = df[c].replace('', '0')
            try:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
            except Exception:
                # If conversion fails, keep as 0
                df[c] = 0
    
    return df


app = Flask(__name__, template_folder=str(MODELS_DIR), static_folder=str(MODELS_DIR))

# Add a route to serve static files from the deploy directory
@app.route('/deploy/<path:filename>')
def serve_static(filename):
    return send_from_directory(MODELS_DIR, filename)

@app.route("/")
def index():
    models = list_available_models()
    return render_template("index.html", models=models)

@app.get("/metadata")
def get_metadata():
    try:
        # Add debugging information
        print(f"Current working directory: {Path.cwd()}")
        print(f"APP_DIR: {APP_DIR}")
        print(f"MODELS_DIR: {MODELS_DIR}")
        print(f"MODELS_DIR exists: {MODELS_DIR.exists()}")
        
        if not MODELS_DIR.exists():
            return jsonify({"error": f"Models directory not found at {MODELS_DIR}"}), 400
        
        # List all files in MODELS_DIR for debugging
        all_files = list(MODELS_DIR.iterdir())
        print(f"Files in MODELS_DIR: {[f.name for f in all_files]}")
        
        metadata = load_metadata()
        models = list_available_models()
        
        print(f"Available models: {list(models.keys())}")
        
        # Define the preferred order for models
        models_order = ['RF', 'XGB', 'Ada', 'DT', 'NB', 'MLP', 'KNN', 'SVM']
        
        # Sort models according to preferred order
        available_models = list(models.keys())
        sorted_models = []
        
        # Add models in preferred order if they exist
        for model in models_order:
            if model in available_models:
                sorted_models.append(model)
        
        # Add any remaining models that weren't in the preferred order
        for model in available_models:
            if model not in sorted_models:
                sorted_models.append(model)
        
        return jsonify({
            "models": sorted_models,
            "feature_names": metadata.get("feature_names", []),
            "target_classes": [int(c) for c in metadata.get("target_classes", [])],
            "training_date": metadata.get("training_date"),
            "n_features": int(metadata.get("n_features", 0)),
            "n_samples": int(metadata.get("n_samples", 0)),
        })
    except Exception as e:
        print(f"Error in get_metadata: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400

@app.post("/predict")
def predict():
    try:
        body = request.get_json(force=True)
        model_name = body.get("model_name")
        features = body.get("features")
        if not model_name:
            return jsonify({"error": "'model_name' is required"}), 400
        if features is None:
            return jsonify({"error": "'features' is required"}), 400

        metadata = load_metadata()
        feature_names = metadata.get("feature_names", [])
        model = load_model(model_name)

        # Add debugging
        #print(f"Received payload: {body}")
        #print(f"Feature names expected: {feature_names}")
        
        X = coerce_features_to_dataframe(features, feature_names)
        #print(f"Processed features shape: {X.shape}")
        #print(f"Processed features:\n{X}")
        
        preds = model.predict(X)
        # Convert numpy types to Python types for JSON serialization
        result: Dict[str, Any] = {"predictions": [int(p) for p in preds]}
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
            # If probabilistic output is a list of arrays (e.g., OVR), normalize to list
            try:
                probs_list = probs.tolist()
            except Exception:
                probs_list = [p.tolist() for p in probs]
            # Convert numpy types to Python types
            if isinstance(probs_list, list):
                probs_list = [[float(p) for p in row] for row in probs_list]
            result["probabilities"] = probs_list
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/debug")
def debug():
    if not MODELS_DIR.exists():
        return {"deploy_exists": False}
    files = [f.name for f in MODELS_DIR.iterdir()]
    return {"deploy_exists": True, "files": files}

if __name__ == "__main__":
    # Default to port 4000
    app.run(host="0.0.0.0", port=4000, debug=True)
