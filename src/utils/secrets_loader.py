import json
from pathlib import Path

class Secrets:
    def __init__(self, path="config_secrets.json"):
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"❌ No se encuentra {path}. Crea config_secrets.json en la raíz del proyecto."
            )

        with open(p, "r") as f:
            self.data = json.load(f)

    # Acceso a Qiskit
    @property
    def qiskit_api_key(self):
        return self.data["qiskit"]["api_key"]


    @property
    def qiskit_instance(self):
        return self.data["qiskit"]["instance"]

    # Backend HW
    @property
    def hardware_backend(self):
        return self.data["hardware"]["backend_name"]
