from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# =============================================================================
# Logging Configuration
# =============================================================================

LOGGER_NAME = "model_registry"
logger = logging.getLogger(LOGGER_NAME)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# =============================================================================
# Custom Exceptions
# =============================================================================

class RegistryError(Exception):
    """Base exception for model registry errors."""


class RegistryNotFoundError(RegistryError):
    """Raised when the registry root or disease folder is not found."""


class DiseaseNotRegisteredError(RegistryError):
    """Raised when a disease is not registered in the model registry."""


class ArtifactMissingError(RegistryError):
    """Raised when one or more required artifacts are missing."""


class ArtifactLoadError(RegistryError):
    """Raised when artifact loading fails."""


class ValidationError(RegistryError):
    """Raised when registry or input validation fails."""


# =============================================================================
# Registry Constants
# =============================================================================

REQUIRED_STRUCTURE = {
    "data_clean": [
        "schema.json",
        "feature_info.json",
        "data_cleaning_report.json",
    ],
    "eda": [
        "eda_summary.json",
        "eda_report.txt",
    ],
    "train": [
        "model.pkl",
        "preprocessor.pkl",
        "selected_features.json",
        "threshold.json",
        "training_config.json",
        "training_summary.json",
        "model_metadata.json",
        "train_metrics.json",
    ],
    "evaluate": [
        "evaluation_metrics.json",
        "classification_report.json",
        "threshold_analysis.csv",
        "error_analysis.json",
    ],
    "shap": [
        "global_feature_impact.csv",
        "local_explanations.json",
        "explainability_summary.json",
    ],
}

OPTIONAL_STRUCTURE = {
    "evaluate": [
        "confusion_matrix.png",
        "roc_curve.png",
        "precision_recall_curve.png",
        "calibration_curve.png",
    ],
    "feature_importance": [
        "feature_importance.csv",
        "feature_importance.json",
        "feature_importance.png",
    ],
    "shap": [
        "shap_summary.png",
        "shap_bar.png",
    ],
}


# =============================================================================
# Helper Functions
# =============================================================================

def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON file safely."""
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        raise ArtifactLoadError(f"Failed to load JSON: {path}") from exc


def load_pickle(path: Path) -> Any:
    """Load pickle file safely."""
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except Exception as exc:
        raise ArtifactLoadError(f"Failed to load pickle artifact: {path}") from exc


def safe_read_text(path: Path) -> str:
    """Read text file safely."""
    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:
        raise ArtifactLoadError(f"Failed to read text file: {path}") from exc


def normalize_folder_name(path: Path) -> Optional[Path]:
    """
    Resolve feature importance folder naming mismatch if present.

    Supports:
    - feature_importance
    - feature importance
    """
    if path.exists():
        return path

    alt = path.parent / path.name.replace("_", " ")
    if alt.exists():
        return alt

    alt2 = path.parent / path.name.replace(" ", "_")
    if alt2.exists():
        return alt2

    return None


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class DiseaseArtifactPaths:
    disease: str
    root: Path
    data_clean: Path
    eda: Path
    train: Path
    evaluate: Path
    feature_importance: Optional[Path]
    shap: Path

    def as_dict(self) -> Dict[str, str]:
        return {
            "disease": self.disease,
            "root": str(self.root),
            "data_clean": str(self.data_clean),
            "eda": str(self.eda),
            "train": str(self.train),
            "evaluate": str(self.evaluate),
            "feature_importance": str(self.feature_importance) if self.feature_importance else None,
            "shap": str(self.shap),
        }


@dataclass
class DiseaseRegistryEntry:
    disease: str
    root: Path
    artifact_paths: DiseaseArtifactPaths
    schema: Dict[str, Any] = field(default_factory=dict)
    feature_info: Dict[str, Any] = field(default_factory=dict)
    data_cleaning_report: Dict[str, Any] = field(default_factory=dict)
    eda_summary: Dict[str, Any] = field(default_factory=dict)
    training_config: Dict[str, Any] = field(default_factory=dict)
    training_summary: Dict[str, Any] = field(default_factory=dict)
    model_metadata: Dict[str, Any] = field(default_factory=dict)
    train_metrics: Dict[str, Any] = field(default_factory=dict)
    threshold_config: Dict[str, Any] = field(default_factory=dict)
    selected_features_payload: Dict[str, Any] = field(default_factory=dict)
    evaluation_metrics: Dict[str, Any] = field(default_factory=dict)
    classification_report: Dict[str, Any] = field(default_factory=dict)
    error_analysis: Dict[str, Any] = field(default_factory=dict)
    feature_importance_summary: Dict[str, Any] = field(default_factory=dict)
    explainability_summary: Dict[str, Any] = field(default_factory=dict)

    def get_feature_info(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "feature_info": self.feature_info,
            "selected_features": self.get_selected_features(),
            "target_column": self.get_target_column(),
        }


    def get_selected_features(self) -> List[str]:
        keys_to_try = [
            "selected_features",
            "features",
            "base_and_engineered_features",
            "input_features_before_preprocessing",
        ]
        for key in keys_to_try:
            value = self.selected_features_payload.get(key)
            if isinstance(value, list):
                return value
        return []

    def get_threshold(self) -> Optional[float]:
        for key in ["threshold", "selected_threshold", "optimal_threshold"]:
            value = self.threshold_config.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        return None

    def get_target_column(self) -> Optional[str]:
        for payload in [
            self.model_metadata,
            self.training_config,
            self.schema,
            self.selected_features_payload,
        ]:
            if isinstance(payload, dict) and payload.get("target_column"):
                return payload["target_column"]
        return None

    def get_model_name(self) -> Optional[str]:
        for key in ["model_name", "model_class", "selected_model"]:
            if key in self.model_metadata:
                return self.model_metadata.get(key)
            if key in self.threshold_config:
                return self.threshold_config.get(key)
            if key in self.training_config:
                return self.training_config.get(key)
        return None

    def to_summary(self) -> Dict[str, Any]:
        return {
            "disease": self.disease,
            "root": str(self.root),
            "target_column": self.get_target_column(),
            "model_name": self.get_model_name(),
            "selected_features_count": len(self.get_selected_features()),
            "threshold": self.get_threshold(),
            "artifact_paths": self.artifact_paths.as_dict(),
        }


@dataclass
class LoadedInferenceBundle:
    disease: str
    model: Any
    preprocessor: Any
    threshold: Optional[float]
    selected_features: List[str]
    target_column: Optional[str]
    metadata: Dict[str, Any]


# =============================================================================
# Model Registry
# =============================================================================

class ModelRegistry:
    """
    Unified model registry for all disease prediction models.

    Main Responsibilities
    ---------------------
    - Discover disease entries
    - Validate registry completeness
    - Load inference artifacts
    - Expose metadata for downstream predictor services
    - Generate manifest / summary for monitoring and debugging
    """

    def __init__(self, registry_root: str | Path):
        self.registry_root = Path(registry_root).resolve()

        if not self.registry_root.exists():
            raise RegistryNotFoundError(
                f"Model registry root does not exist: {self.registry_root}"
            )

        self._entries: Dict[str, DiseaseRegistryEntry] = {}

        self._inference_cache: Dict[str, LoadedInferenceBundle] = {}

        logger.info("Initialized model registry at: %s", self.registry_root)

    # -------------------------------------------------------------------------
    # Discovery
    # -------------------------------------------------------------------------

    def discover(self, strict: bool = True) -> Dict[str, DiseaseRegistryEntry]:
        """Discover all disease folders and register valid entries."""
        logger.info("Discovering disease registries...")
        self._entries.clear()

        for item in sorted(self.registry_root.iterdir()):
            if not item.is_dir():
                continue

            disease = item.name
            try:
                entry = self._build_entry(disease)
                self._entries[disease] = entry
                logger.info("Registered disease: %s", disease)
            except RegistryError as exc:
                if strict:
                    raise
                logger.warning("Skipping disease '%s': %s", disease, exc)

        logger.info("Discovery completed. Registered diseases: %s", list(self._entries.keys()))
        return self._entries

    def list_diseases(self) -> List[str]:
        """Return sorted list of registered diseases."""
        return sorted(self._entries.keys())

    def is_registered(self, disease: str) -> bool:
        """Check whether a disease is registered."""
        return disease in self._entries

    # -------------------------------------------------------------------------
    # Entry Building
    # -------------------------------------------------------------------------

    def _build_entry(self, disease: str) -> DiseaseRegistryEntry:
        disease_root = self.registry_root / disease
        if not disease_root.exists():
            raise DiseaseNotRegisteredError(f"Disease folder not found: {disease}")

        data_clean_dir = disease_root / "data_clean"
        eda_dir = disease_root / "eda"
        train_dir = disease_root / "train"
        evaluate_dir = disease_root / "evaluate"
        shap_dir = disease_root / "shap"

        feature_importance_dir = normalize_folder_name(disease_root / "feature_importance")

        artifact_paths = DiseaseArtifactPaths(
            disease=disease,
            root=disease_root,
            data_clean=data_clean_dir,
            eda=eda_dir,
            train=train_dir,
            evaluate=evaluate_dir,
            feature_importance=feature_importance_dir,
            shap=shap_dir,
        )

        self._validate_required_structure(disease_root)

        entry = DiseaseRegistryEntry(
            disease=disease,
            root=disease_root,
            artifact_paths=artifact_paths,
            schema=load_json(data_clean_dir / "schema.json"),
            feature_info=load_json(data_clean_dir / "feature_info.json"),
            data_cleaning_report=load_json(data_clean_dir / "data_cleaning_report.json"),
            eda_summary=load_json(eda_dir / "eda_summary.json"),
            training_config=load_json(train_dir / "training_config.json"),
            training_summary=load_json(train_dir / "training_summary.json"),
            model_metadata=load_json(train_dir / "model_metadata.json"),
            train_metrics=load_json(train_dir / "train_metrics.json"),
            threshold_config=load_json(train_dir / "threshold.json"),
            selected_features_payload=load_json(train_dir / "selected_features.json"),
            evaluation_metrics=load_json(evaluate_dir / "evaluation_metrics.json"),
            classification_report=load_json(evaluate_dir / "classification_report.json"),
            error_analysis=load_json(evaluate_dir / "error_analysis.json"),
            feature_importance_summary=self._load_optional_feature_importance(feature_importance_dir),
            explainability_summary=load_json(shap_dir / "explainability_summary.json"),
        )

        return entry

    def _validate_required_structure(self, disease_root: Path) -> None:
        missing: List[str] = []

        for folder, files in REQUIRED_STRUCTURE.items():
            folder_path = disease_root / folder

            if not folder_path.exists():
                missing.append(str(folder_path))
                continue

            for filename in files:
                file_path = folder_path / filename
                if not file_path.exists():
                    missing.append(str(file_path))

        if missing:
            raise ArtifactMissingError(
                f"Missing required artifacts in '{disease_root.name}': {missing}"
            )

        # ✅ ADD THIS BLOCK
        for folder, files in OPTIONAL_STRUCTURE.items():
            folder_path = disease_root / folder
            if folder_path.exists():
                for filename in files:
                    file_path = folder_path / filename
                    if not file_path.exists():
                        logger.warning("Optional artifact missing: %s", file_path)

    def _load_optional_feature_importance(self, feature_importance_dir: Optional[Path]) -> Dict[str, Any]:
        """Load optional feature importance summary if available."""
        if feature_importance_dir is None or not feature_importance_dir.exists():
            return {}

        json_path = feature_importance_dir / "feature_importance.json"
        if json_path.exists():
            return load_json(json_path)
        return {}

    # -------------------------------------------------------------------------
    # Accessors
    # -------------------------------------------------------------------------

    def get_entry(self, disease: str) -> DiseaseRegistryEntry:
        """Get registry entry for disease."""
        if disease not in self._entries:
            raise DiseaseNotRegisteredError(
                f"Disease '{disease}' is not registered. "
                f"Available diseases: {self.list_diseases()}"
            )
        return self._entries[disease]

    def get_registry_summary(self) -> Dict[str, Any]:
        """Return high-level summary of all registered models."""
        return {
            "registry_root": str(self.registry_root),
            "generated_at": utc_now_iso(),
            "registered_diseases": self.list_diseases(),
            "total_diseases": len(self._entries),
            "entries": [entry.to_summary() for entry in self._entries.values()],
        }

    def save_registry_summary(self, output_path: str | Path) -> Path:
        """Save registry summary JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        summary = self.get_registry_summary()
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)

        logger.info("Registry summary saved to: %s", output_path)
        return output_path

    # -------------------------------------------------------------------------
    # Health Checks
    # -------------------------------------------------------------------------

    def validate_registry(self) -> Dict[str, Any]:
        report = {
            "registry_root": str(self.registry_root),
            "validated_at": utc_now_iso(),
            "overall_status": "healthy",
            "diseases": {},
        }

        for disease, entry in self._entries.items():
            issues = []

            selected_features = entry.get_selected_features()

            if not selected_features:
                issues.append("Selected features list is empty.")

            # ✅ Added strict validation
            if len(selected_features) < 3:
                issues.append("Too few selected features (possible training issue).")

            threshold = entry.get_threshold()

            if threshold is None:
                issues.append("Threshold missing.")
            elif not (0.0 <= threshold <= 1.0):
                issues.append(f"Invalid threshold: {threshold}")

            if not entry.get_target_column():
                issues.append("Target column missing.")

            status = "healthy" if not issues else "warning"
            if issues:
                report["overall_status"] = "warning"

            report["diseases"][disease] = {
                "status": status,
                "issues": issues,
                "threshold": threshold,
                "selected_features_count": len(selected_features),
            }

        return report


    def save_validation_report(self, output_path: str | Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        report = self.validate_registry()
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)

        logger.info("Registry validation report saved to: %s", output_path)
        return output_path

    # -------------------------------------------------------------------------
    # Inference Loading
    # -------------------------------------------------------------------------

    def load_inference_bundle(self, disease: str) -> LoadedInferenceBundle:
        if disease in self._inference_cache:
            return self._inference_cache[disease]

        entry = self.get_entry(disease)

        model_path = entry.artifact_paths.train / "model.pkl"
        preprocessor_path = entry.artifact_paths.train / "preprocessor.pkl"

        if not model_path.exists():
            raise ArtifactMissingError(f"Model file missing: {model_path}")

        if not preprocessor_path.exists():
            raise ArtifactMissingError(f"Preprocessor file missing: {preprocessor_path}")

        model = load_pickle(model_path)
        preprocessor = load_pickle(preprocessor_path)

        bundle = LoadedInferenceBundle(
            disease=disease,
            model=model,
            preprocessor=preprocessor,
            threshold=entry.get_threshold(),
            selected_features=entry.get_selected_features(),
            target_column=entry.get_target_column(),
            metadata={
                "model_metadata": entry.model_metadata,
                "training_config": entry.training_config,
                "training_summary": entry.training_summary,
                "evaluation_metrics": entry.evaluation_metrics,
                "explainability_summary": entry.explainability_summary,
            },
        )

        self._inference_cache[disease] = bundle
        return bundle


    def prepare_prediction_context(self, disease: str) -> Dict[str, Any]:
        bundle = self.load_inference_bundle(disease)

        return {
            "model": bundle.model,
            "preprocessor": bundle.preprocessor,
            "threshold": bundle.threshold,
            "input_features": self.get_required_input_features(disease),
            "selected_features": bundle.selected_features,
            "target_column": bundle.target_column,
            "metadata": bundle.metadata,
        }


    # -------------------------------------------------------------------------
    # Input Validation for Predictor Layer
    # -------------------------------------------------------------------------

    def get_required_input_features(self, disease: str) -> List[str]:
        """
        Return RAW input features required for prediction (before preprocessing).
        """

        entry = self.get_entry(disease)

        payload = entry.selected_features_payload

        # 1. Best source → raw input features
        for key in [
            "input_features_before_preprocessing",
            "all_features",
            "base_and_engineered_features",
            "input_feature_names",
        ]:
            value = payload.get(key)
            if isinstance(value, list) and len(value) > 0:
                return value

        # 2. fallback → schema
        schema = entry.schema
        for key in ["feature_columns", "features"]:
            value = schema.get(key)
            if isinstance(value, list):
                return value

        # 3. fallback → schema columns
        columns = schema.get("columns")
        target_col = entry.get_target_column()

        if isinstance(columns, list):
            normalized = []
            for col in columns:
                if isinstance(col, dict):
                    name = col.get("name")
                    if name and name != target_col:
                        normalized.append(name)
                elif isinstance(col, str) and col != target_col:
                    normalized.append(col)
            return normalized
        raise ValidationError(
            f"Could not determine required input features for disease '{disease}'."
        )

    def validate_input_payload(
        self,
        disease: str,
        payload: Dict[str, Any],
        strict: bool = True,
    ) -> Dict[str, Any]:
        """
        Validate input payload against registry-required features.

        Parameters
        ----------
        disease : str
            Disease name.
        payload : Dict[str, Any]
            Incoming prediction payload.
        strict : bool
            If True, raise error on missing fields.
            If False, return report only.

        Returns
        -------
        Dict[str, Any]
            Validation report.
        """
        if not isinstance(payload, dict):
            raise ValidationError("Prediction payload must be a dictionary.")

        required_features = self.get_required_input_features(disease)
        missing = [
            f for f in required_features
            if f not in payload or payload[f] is None or payload[f] == ""
        ]
        extra = [f for f in payload.keys() if f not in required_features]

        report = {
            "disease": disease,
            "required_feature_count": len(required_features),
            "provided_feature_count": len(payload),
            "missing_features": missing,
            "extra_features": extra,
            "is_valid": len(missing) == 0,
        }

        if strict and missing:
            raise ValidationError(
                f"Missing required fields for '{disease}': {missing}"
            )

        return report

    # -------------------------------------------------------------------------
    # Deployment / Manifest Utilities
    # -------------------------------------------------------------------------

    def build_manifest(self) -> Dict[str, Any]:
        """
        Build deployment manifest for all registered diseases.
        Useful for APIs, monitoring, and release packaging.
        """
        manifest = {
            "registry_name": "RiskLens Model Registry",
            "registry_root": str(self.registry_root),
            "generated_at": utc_now_iso(),
            "version": "1.0.0",
            "diseases": {},
        }

        for disease, entry in self._entries.items():
            manifest["diseases"][disease] = {
                "disease": disease,
                "target_column": entry.get_target_column(),
                "model_name": entry.get_model_name(),
                "selected_features": entry.get_selected_features(),
                "selected_features_count": len(entry.get_selected_features()),
                "threshold": entry.get_threshold(),
                "paths": entry.artifact_paths.as_dict(),
            }

        return manifest

    def save_manifest(self, output_path: str | Path) -> Path:
        """Save deployment manifest JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        manifest = self.build_manifest()
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=4)

        logger.info("Registry manifest saved to: %s", output_path)
        return output_path


# =============================================================================
# Convenience Functions
# =============================================================================

def create_registry(registry_root: str | Path, strict: bool = True) -> ModelRegistry:
    """Create and discover model registry."""
    registry = ModelRegistry(registry_root=registry_root)
    registry.discover(strict=strict)
    return registry


def load_registry_entry(registry_root: str | Path, disease: str) -> DiseaseRegistryEntry:
    """Create registry and load one disease entry."""
    registry = create_registry(registry_root=registry_root, strict=True)
    return registry.get_entry(disease)


def load_inference_assets(registry_root: str | Path, disease: str) -> LoadedInferenceBundle:
    """Create registry and load inference assets for one disease."""
    registry = create_registry(registry_root=registry_root, strict=True)
    return registry.load_inference_bundle(disease)


# =============================================================================
# CLI / Standalone Execution
# =============================================================================


if __name__ == "__main__":
    # Example local usage:
    #
    # python model_registry.py
    #
    # Update this path based on your project structure if needed.
    default_registry_root = Path("model_registry")

    try:
        registry = create_registry(default_registry_root, strict=False)

        print("\n" + "=" * 90)
        print("RISKLENS MODEL REGISTRY SUMMARY")
        print("=" * 90)
        print(json.dumps(registry.get_registry_summary(), indent=4))

        print("\n" + "=" * 90)
        print("RISKLENS MODEL REGISTRY VALIDATION")
        print("=" * 90)
        print(json.dumps(registry.validate_registry(), indent=4))

    except Exception as exc:
        logger.exception("Registry execution failed: %s", exc)
        raise
