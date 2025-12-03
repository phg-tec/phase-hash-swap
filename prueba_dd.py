import math
import numpy as np
from sklearn.cluster import KMeans
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import DiagonalGate
from qiskit_aer import AerSimulator


# -----------------------------------------------------
# UTILIDADES BÁSICAS
# -----------------------------------------------------

def next_power_of_two(m: int) -> int:
    return 1 if m <= 1 else 1 << (m - 1).bit_length()

def pad_to_power_of_two(values):
    L = len(values)
    m = next_power_of_two(L)
    return values if m == L else np.concatenate([values, np.zeros(m - L)])

def cos_sim(u, v):
    u = np.asarray(u, float)
    v = np.asarray(v, float)
    return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))

def generate_correlated_gaussian_pair(d, rho, rng):
    cov = np.array([[1.0, rho], [rho, 1.0]])
    L = np.linalg.cholesky(cov)
    z = rng.normal(size=(2, d))
    xy = L @ z
    return xy[0], xy[1]


# -----------------------------------------------------
# K-MEANS DISCRETIZATION (VALUES = CENTERS)
# -----------------------------------------------------

def kmeans_to_centers(x, K=2):
    x = np.asarray(x, float)
    z = x.reshape(-1, 1)
    km = KMeans(n_clusters=K, n_init=10, random_state=123).fit(z)
    labels = km.labels_
    centers = km.cluster_centers_.flatten()
    return centers[labels], centers


# -----------------------------------------------------
# REAL → COMPLEX UNIT CIRCLE ENCODING
# -----------------------------------------------------

def real_to_unit_complex(c):
    """
    Mapea c ∈ R a un complejo de módulo 1:

      si c >= 0:  z =  c + i*sqrt(1-c^2)
      si c <  0:  z =  c - i*sqrt(1-c^2)

    De esta forma:
      - el signo de la parte real coincide con el centro
      - el signo de la parte imaginaria también sigue el signo del centro
    """
    c = float(c)
    c = max(min(c, 1.0), -1.0)   # saturamos a [-1,1] por seguridad
    imag = math.sqrt(max(0.0, 1.0 - c*c))

    if c >= 0:
        return c + 1j*imag      # cuadrante superior derecho
    else:
        return c - 1j*imag      # cuadrante inferior izquierdo

def encode_vector_real_to_complex(xvals):
    return np.array([real_to_unit_complex(x) for x in xvals], complex)


# -----------------------------------------------------
# BUILD PHASE STATE
# -----------------------------------------------------

def build_phase_state(complex_vals):
    complex_vals = np.asarray(complex_vals, complex)
    m = len(complex_vals)
    n = int(math.log2(m))
    if 2**n != m:
        raise ValueError("Length must be power of 2")

    qr = QuantumRegister(n, "data")
    qc = QuantumCircuit(qr)

    qc.h(qr)
    qc.append(DiagonalGate(complex_vals.tolist()), qr[:])
    return qc


# -----------------------------------------------------
# SWAP TEST
# -----------------------------------------------------

def build_swap_test_circuit(x_vals, y_vals):
    x_vals = pad_to_power_of_two(x_vals)
    y_vals = pad_to_power_of_two(y_vals)

    m = len(x_vals)
    n = int(math.log2(m))

    anc = QuantumRegister(1, "anc")
    qa  = QuantumRegister(n, "a")
    qb  = QuantumRegister(n, "b")
    c   = ClassicalRegister(1, "c")

    qc = QuantumCircuit(anc, qa, qb, c)

    qc.compose(build_phase_state(x_vals), qa, inplace=True)
    qc.compose(build_phase_state(y_vals), qb, inplace=True)

    qc.h(anc)
    for i in range(n):
        qc.cswap(anc, qa[i], qb[i])
    qc.h(anc)
    qc.measure(anc, c)

    return qc


def run_swap(x_vals, y_vals, shots=4096, backend=None):
    if backend is None:
        backend = AerSimulator()

    qc = build_swap_test_circuit(x_vals, y_vals)
    tqc = transpile(qc, backend)
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts(tqc)
    return counts.get("0", 0) / shots


# -----------------------------------------------------
# EXPERIMENTO FINAL
# -----------------------------------------------------

def run_kmeans_centers_experiment(
    d=512,
    shots=4096,
    seed=123
):
    rng = np.random.default_rng(seed)
    backend = AerSimulator(seed_simulator=seed)

    rho_list = [-0.90, -0.50, -0.25, 0.00, 0.25, 0.50, 0.90]
    K=2
    print("\n=== KMEANS(2) → COMPLEX PHASE ENCODING → SWAP ===")
    print(f"d = {d}, shots = {shots}\n")
    print("rho | cos_real_centers |  p0_hat  | cos_quantum")
    print("---------------------------------------------------")

    for rho in rho_list:

        # datos correlacionados
        x_real, y_real = generate_correlated_gaussian_pair(d, rho, rng)

        # discretización K-means(2)
        x_disc, centers_x = kmeans_to_centers(x_real, K)
        y_disc, centers_y = kmeans_to_centers(y_real, K)
        print(centers_x, centers_y)
        # coseno sobre los valores-centro
        cos_r = cos_sim(x_disc, y_disc)

        # encoding complejo en el círculo unitario
        x_phase = encode_vector_real_to_complex(x_disc)
        y_phase = encode_vector_real_to_complex(y_disc)

        # swap
        p0 = run_swap(x_phase, y_phase, shots, backend)

        # estimador tipo AE
        overlap = np.sign(np.dot(x_disc, y_disc))*math.sqrt(max(0.0, 2*p0 - 1.0))

        print(f"{rho:+.2f} | {cos_r:+.3f}          | {p0:.6f} | {overlap:+.3f}")


if __name__ == "__main__":
    run_kmeans_centers_experiment()
