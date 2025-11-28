import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA

# ----------------------------------------
#   ITQ IMPLEMENTATION
# ----------------------------------------

def itq_rotation(Z, num_iters=50, seed=123):
    np.random.seed(seed)
    N, m = Z.shape

    R = np.random.randn(m, m)
    R, _ = np.linalg.qr(R)

    for _ in range(num_iters):
        B = np.sign(Z @ R)
        C = B.T @ Z
        U, _, Vt = np.linalg.svd(C)
        R = Vt.T @ U.T

    return R


def train_pca_itq(X, m, itq_iters=50):
    N, d = X.shape
    print(f"Training PCA+ITQ for n={d} → m={m}")

    pca = PCA(n_components=m, svd_solver="randomized")
    Z = pca.fit_transform(X)
    W = pca.components_.T

    R = itq_rotation(Z, num_iters=itq_iters)

    return W, R


# ----------------------------------------
#   MASTER SCRIPT: TRAIN ALL (n,m)
# ----------------------------------------

if __name__ == "__main__":

    # Reduction pairs you requested
    reduction_pairs = {
        64:   [16, 32],
        128:  [32, 64],
        256:  [64, 128],
        512:  [128, 256],
        1024: [256, 512]
    }

    N = 20000   # training samples for each n
    results = {}

    print("\n=== Starting full PCA+ITQ training for all (n → m) ===\n")

    for n, m_list in reduction_pairs.items():

        print(f"\n---------- GENERATING DATA FOR n = {n} ----------")
        X = np.random.randn(N, n)   # <<<< GENERATES dataset for each n

        for m in tqdm(m_list, desc=f"Training (n={n})"):

            W, R = train_pca_itq(X, m)
            results[(n, m)] = (W, R)

            np.save(f"experiments/itq/matrices/W_pca_{n}_to_{m}.npy", W)
            np.save(f"experiments/itq/matrices/R_itq_{n}_to_{m}.npy", R)

            print(f"Saved W_pca_{n}_to_{m}.npy and R_itq_{n}_to_{m}.npy")

    print("\n=== All (n → m) reductions completed successfully! ===\n")
