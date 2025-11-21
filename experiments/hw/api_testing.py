from qiskit_ibm_runtime import QiskitRuntimeService
from src.utils.secrets_loader import Secrets

# Cargar claves
sec = Secrets()

# =====================================================
# Configuración IBM Quantum
# =====================================================
service = QiskitRuntimeService(
    channel="ibm_quantum_platform",
    token=sec.qiskit_api_key,
    instance=sec.qiskit_instance
)
backend = service.backend(sec.hardware_backend)
print("Usando backend:", backend.name)
jobs = service.jobs(limit=5)
for j in jobs:
    print(j.job_id(), j.backend().name, j.status())