"""
scripts/update_services_metrics.py
"""
from pathlib import Path

def update_sensor_service():
    p = Path("src/services/sensor_forecast_service.py")
    content = p.read_text(encoding="utf-8")
    old_block = """            "metrics": {
                "pm25_mae": self.saved_metrics.get("pm25", {}).get("MAE", 10.57),
                "pm25_rmse": self.saved_metrics.get("pm25", {}).get("RMSE", 18.58),
                "pm25_r2": self.saved_metrics.get("pm25", {}).get("R2", 0.9492),
                "pm10_mae": self.saved_metrics.get("pm10", {}).get("MAE", 18.86),
                "pm10_rmse": self.saved_metrics.get("pm10", {}).get("RMSE", 32.20),
                "pm10_r2": self.saved_metrics.get("pm10", {}).get("R2", 0.9301),
                "evaluation_cohort": "404,431 Test Sequences (2015-2020 CPCB Benchmark)",
            },"""
    new_block = """            "metrics": {
                "pm25_mae": (self.saved_metrics.get("PM2.5") or self.saved_metrics.get("pm25") or {}).get("MAE", 10.57),
                "pm25_rmse": (self.saved_metrics.get("PM2.5") or self.saved_metrics.get("pm25") or {}).get("RMSE", 18.58),
                "pm25_r2": (self.saved_metrics.get("PM2.5") or self.saved_metrics.get("pm25") or {}).get("R2", 0.9492),
                "pm10_mae": (self.saved_metrics.get("PM10") or self.saved_metrics.get("pm10") or {}).get("MAE", 18.86),
                "pm10_rmse": (self.saved_metrics.get("PM10") or self.saved_metrics.get("pm10") or {}).get("RMSE", 32.20),
                "pm10_r2": (self.saved_metrics.get("PM10") or self.saved_metrics.get("pm10") or {}).get("R2", 0.9301),
                "evaluation_cohort": "404,431 Test Sequences (2015-2020 CPCB Benchmark)",
            },"""
    assert old_block in content, "old_block not found in sensor_forecast_service.py"
    p.write_text(content.replace(old_block, new_block), encoding="utf-8")
    print("Updated sensor_forecast_service.py")

def update_image_service():
    p = Path("src/services/image_estimation_service.py")
    content = p.read_text(encoding="utf-8")
    old_load = """    def _load_metrics(self):
        # Metrics from validation audit
        self.saved_metrics = {
            "MAE": 60.99,
            "RMSE": 88.20,
            "R2": -0.2023,
            "evaluation_cohort": "Held-Out Test Cohort (N = 97 matched images)",
        }"""
    new_load = """    def _load_metrics(self):
        metrics_file = ROOT_DIR / "experiments/multimodal_fusion/validation_v2/metrics.json"
        if metrics_file.exists():
            import json
            with open(metrics_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            t_m = data.get("test_metrics", {}).get("image_only_resnet50", {})
            self.saved_metrics = {
                "MAE": float(t_m.get("MAE", 64.99)),
                "RMSE": float(t_m.get("RMSE", 94.59)),
                "R2": float(t_m.get("R2", -0.3828)),
                "evaluation_cohort": "Held-Out Test Cohort (N = 97 matched images, zero group leakage)",
            }
        else:
            self.saved_metrics = {
                "MAE": 64.99,
                "RMSE": 94.59,
                "R2": -0.3828,
                "evaluation_cohort": "Held-Out Test Cohort (N = 97 matched images, zero group leakage)",
            }"""
    assert old_load in content, "old_load not found in image_estimation_service.py"
    p.write_text(content.replace(old_load, new_load), encoding="utf-8")
    print("Updated image_estimation_service.py")

if __name__ == "__main__":
    update_sensor_service()
    update_image_service()
