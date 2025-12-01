#!/usr/bin/env python
"""
Demo estratificado tipo SWAP:
- Generar dos vectores con un coseno dado
- Discretizarlos con k-medias en k centros
- Particionar las coordenadas en grupos (p,q) SIN solapamiento
- Estimar el coseno global a partir de cosenos locales de grupo
  * modo 'classical': cosenos locales clásicos
  * modo 'quantum'  : cosenos locales estimados vía un stub de SWAP/PES
"""

import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
from typing import Dict, Tuple


# ------------------------------------------------------------
# 1) Generar vectores con un coseno objetivo
# ------------------------------------------------------------
def generate_pair_with_cosine(d, cos_target, rng=None):
    """
    Genera dos vectores x, y en R^d tales que cos(x,y) ~= cos_target.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Vector aleatorio unitario u
    u = rng.normal(size=d)
    u /= np.linalg.norm(u)

    # Vector v ortogonal a u
    v = rng.normal(size=d)
    v -= u * np.dot(u, v)
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-12:
        # caso degenerado muy raro → reintentamos
        return generate_pair_with_cosine(d, cos_target, rng)
    v /= v_norm

    # Construimos y con el ángulo deseado
    sin_theta = np.sqrt(max(0.0, 1.0 - cos_target**2))
    x = u
    y = cos_target * u + sin_theta * v
    return x, y


# ------------------------------------------------------------
# 2) Discretización con k-medias
# ------------------------------------------------------------
def discretize_with_kmeans(a, b, k, random_state=0):
    """
    Discretiza dos vectores 1D a y b usando k-medias en 1D.
    Se ajusta un único k-medias sobre todos los valores de a y b,
    y luego se reemplaza cada valor por el centro asignado.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = len(a)

    vals = np.concatenate([a, b])[:, None]  # shape (2d, 1)

    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    km.fit(vals)

    centers = km.cluster_centers_.flatten()  # k centros
    labels = km.labels_
    labels_a = labels[:d]
    labels_b = labels[d:]

    a_disc = centers[labels_a]
    b_disc = centers[labels_b]
    return a_disc, b_disc, centers


# ------------------------------------------------------------
# 3) Partición en grupos (p,q) SIN solapamiento
# ------------------------------------------------------------
def build_partition_groups(a_disc, b_disc, centers):
    """
    Particiona las coordenadas en grupos etiquetados por pares (p,q),
    donde p y q son ÍNDICES de centros (0..k-1) y:

        p = idx(centro de a_disc[i])
        q = idx(centro de b_disc[i])

    Para cada coordenada i:
        - calculamos (p,q) según el centro de a[i] y b[i]
        - normalizamos el par como (min(p,q), max(p,q))
        - añadimos i al grupo correspondiente

    Esto define una partición: cada i pertenece a UN solo grupo.

    Devuelve:
      groups[(p,q)] = lista de índices i
      centers_idx -> mapeo valor de centro -> índice
    """
    a_disc = np.asarray(a_disc)
    b_disc = np.asarray(b_disc)
    centers = sorted(list(centers))

    center_to_idx = {c: i for i, c in enumerate(centers)}  # valor -> idx

    groups: Dict[Tuple[int, int], list] = defaultdict(list)

    for i in range(len(a_disc)):
        p = center_to_idx[a_disc[i]]
        q = center_to_idx[b_disc[i]]
        if p <= q:
            key = (p, q)
        else:
            key = (q, p)
        groups[key].append(i)

    return groups, centers, center_to_idx


# ------------------------------------------------------------
# 4) Estimador clásico EXACTO desde los grupos
# ------------------------------------------------------------
def classical_cos_from_groups(a_disc, b_disc, groups):
    """
    Implementa la identidad:

      cos(a,b)
      = [ Σ_{(p,q)} cos_{p,q} * sqrt(Na_{p,q} * Nb_{p,q}) ]
        / sqrt( (Σ Na_{p,q}) (Σ Nb_{p,q}) )

    donde:
      - Na_{p,q} = ||a^(p,q)||^2
      - Nb_{p,q} = ||b^(p,q)||^2
      - cos_{p,q} = cos(a^(p,q), b^(p,q))

    Si cos_{p,q} se calcula clásicamente, el resultado ES EXACTAMENTE
    el coseno de (a_disc, b_disc).
    """
    a_disc = np.asarray(a_disc, float)
    b_disc = np.asarray(b_disc, float)

    Na_g = {}
    Nb_g = {}
    cos_g = {}

    # Cálculo local por grupo
    for key, idxs in groups.items():
        idxs = np.asarray(idxs, int)
        sub_a = a_disc[idxs]
        sub_b = b_disc[idxs]

        na = float(sub_a @ sub_a)
        nb = float(sub_b @ sub_b)
        Na_g[key] = na
        Nb_g[key] = nb

        if na > 0 and nb > 0:
            cos_g[key] = float(sub_a @ sub_b) / np.sqrt(na * nb)
        else:
            cos_g[key] = 0.0

    # Agregación global
    Na = sum(Na_g.values())
    Nb = sum(Nb_g.values())

    num = 0.0
    for key in groups:
        num += cos_g[key] * np.sqrt(Na_g[key] * Nb_g[key])

    cos_hat = num / np.sqrt(Na * Nb)
    return cos_hat, cos_g, Na_g, Nb_g


# ------------------------------------------------------------
# 5) Stub cuántico: aquí se enchufa el SWAP / PES-SWAP
# ------------------------------------------------------------
def estimate_cosine_group_quantum(sub_a, sub_b, method="PES"):
    """
    Stub que representa el 'estimador cuántico' de coseno para un grupo.

    sub_a, sub_b: subvectores (reales) de ese grupo (m_g dimensiones).

    En este stub, devolvemos simplemente el coseno clásico, pero
    conceptualmente aquí es donde:
      - preparas |ψ_a> y |ψ_b> (AE o PES)
      - ejecutas el circuito SWAP o PES-SWAP con shots
      - estimas p0 y lo transformas en cos_hat

    method:
      - 'AE'  -> código futuro de AE-SWAP
      - 'PES' -> código futuro de PES-SWAP
    """
    sub_a = np.asarray(sub_a, float)
    sub_b = np.asarray(sub_b, float)

    na = float(sub_a @ sub_a)
    nb = float(sub_b @ sub_b)
    if na == 0 or nb == 0:
        return 0.0

    # POR AHORA: estimador clásico (para debug)
    cos_classical = float(sub_a @ sub_b) / np.sqrt(na * nb)
    return cos_classical


def stratified_cosine_estimator(a_disc, b_disc, groups, mode="classical"):
    """
    Estimador estratificado del coseno:

      - mode='classical':
          usa classical_cos_from_groups (exacto)

      - mode='quantum':
          sustituye cada cos_{p,q} clásico por un
          cos_{p,q} 'cuántico' obtenido con estimate_cosine_group_quantum
          y luego aplica la MISMA fórmula de agregación.
    """
    a_disc = np.asarray(a_disc, float)
    b_disc = np.asarray(b_disc, float)

    Na_g = {}
    Nb_g = {}
    cos_g = {}

    # 1) computar normas parciales (clásicas)
    for key, idxs in groups.items():
        idxs = np.asarray(idxs, int)
        sub_a = a_disc[idxs]
        sub_b = b_disc[idxs]

        na = float(sub_a @ sub_a)
        nb = float(sub_b @ sub_b)
        Na_g[key] = na
        Nb_g[key] = nb

        if na == 0 or nb == 0:
            cos_g[key] = 0.0
            continue

        if mode == "classical":
            cos_g[key] = float(sub_a @ sub_b) / np.sqrt(na * nb)
        elif mode == "quantum":
            cos_g[key] = estimate_cosine_group_quantum(sub_a, sub_b, method="PES")
        else:
            raise ValueError(f"Modo desconocido: {mode}")

    # 2) agregación global
    Na = sum(Na_g.values())
    Nb = sum(Nb_g.values())

    num = 0.0
    for key in groups:
        num += cos_g[key] * np.sqrt(Na_g[key] * Nb_g[key])

    cos_hat = num / np.sqrt(Na * Nb)
    return cos_hat, cos_g, Na_g, Nb_g


# ------------------------------------------------------------
# 6) Script principal
# ------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Estimador estratificado de coseno tipo SWAP usando grupos (p,q)."
    )
    parser.add_argument("--dim", type=int, default=256,
                        help="Dimensión de los vectores.")
    parser.add_argument("--cos", type=float, default=0.7,
                        help="Coseno objetivo entre los vectores continuos.")
    parser.add_argument("--k", type=int, default=4,
                        help="Número de centros k para k-medias.")
    parser.add_argument("--seed", type=int, default=1234,
                        help="Semilla aleatoria.")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # 1) Generar vectores continuos con coseno objetivo
    x, y = generate_pair_with_cosine(args.dim, args.cos, rng=rng)
    cos_true = float(x @ y) / (np.linalg.norm(x) * np.linalg.norm(y))

    print("=== Generación de vectores continuos ===")
    print(f"Coseno objetivo pedido:     {args.cos:.6f}")
    print(f"Coseno continuo generado:   {cos_true:.6f}")

    # 2) Discretización con k-medias
    a_disc, b_disc, centers = discretize_with_kmeans(x, y, args.k,
                                                     random_state=args.seed)
    cos_disc_direct = float(a_disc @ b_disc) / (
        np.linalg.norm(a_disc) * np.linalg.norm(b_disc)
    )

    print("\n=== Discretización con k-medias ===")
    print(f"k = {args.k}")
    print(f"Centros encontrados: {np.round(centers, 4)}")
    print(f"Coseno directo (vectores discretizados): {cos_disc_direct:.6f}")

    # 3) Partición en grupos (p,q)
    groups, centers_sorted, center_to_idx = build_partition_groups(
        a_disc, b_disc, centers
    )

    # 4) Modo clásico (exacto)
    cos_group_classic, cos_g_classic, Na_g, Nb_g = classical_cos_from_groups(
        a_disc, b_disc, groups
    )

    print("\n=== Agregación por grupos (p,q) - MODO CLÁSICO ===")
    print(f"Coseno por agregación (exacto): {cos_group_classic:.6f}")

    # 5) Modo 'cuántico' (ahora mismo usa el stub que devuelve cos clásico)
    cos_group_quantum, cos_g_quantum, _, _ = stratified_cosine_estimator(
        a_disc, b_disc, groups, mode="quantum"
    )

    print("\n=== Agregación por grupos (p,q) - MODO 'CUÁNTICO' (stub) ===")
    print(f"Coseno estimado (stub): {cos_group_quantum:.6f}")

    print("\nDetalle de grupos no vacíos:")
    for key in sorted(groups.keys()):
        idxs = groups[key]
        print(
            f"  Grupo {key}: n={len(idxs):4d}, "
            f"cos_class={cos_g_classic[key]: .4f}, "
            f"Na={Na_g[key]: .4f}, Nb={Nb_g[key]: .4f}"
        )


if __name__ == "__main__":
    main()
