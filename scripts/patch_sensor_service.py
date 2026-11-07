from pathlib import Path

p = Path("src/services/sensor_forecast_service.py")
text = p.read_text(encoding="utf-8")
text = text.replace("                import numpy as np\n", "")
p.write_text(text, encoding="utf-8")
print("Removed local import numpy as np")
