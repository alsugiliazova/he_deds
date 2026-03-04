#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from Pyfhel import Pyfhel
import matplotlib.pyplot as plt
import numpy as np
import time


def print_results(x_axis_values, results):
    plain = np.array([r["plain_time_ms"] for r in results], dtype=float)
    enc = np.array([r["encrypt_time_ms"] for r in results], dtype=float)
    comp = np.array([r["compute_time_ms"] for r in results], dtype=float)
    dec = np.array([r["decrypt_time_ms"] for r in results], dtype=float)
    he_total = enc + comp + dec

    speedup_e2e = he_total / plain 
    speedup_compute = comp / plain 

    # --- Timing plot: grouped + stacked ---
    fig, ax = plt.subplots(figsize=(10, 5))
    idx = np.arange(len(x_axis_values))
    w = 0.34

    ax.bar(idx - w/2, plain, width=w, label="Plain (no HE)")
    ax.bar(idx + w/2, enc,  width=w, label="HE: encode/encrypt")
    ax.bar(idx + w/2, comp, width=w, bottom=enc,     label="HE: compute")
    ax.bar(idx + w/2, dec,  width=w, bottom=enc+comp,label="HE: decode/decrypt")

    for i in range(len(x_axis_values)):
        top = max(plain[i], he_total[i])
        y = top * 1.05
        ax.text(i, y, f"S_total={speedup_e2e[i]:.2e}\nS_comp={speedup_compute[i]:.2e}",
                ha="center", va="bottom", fontsize=9)
    
    ymax = max(np.max(plain), np.max(he_total))
    ax.set_ylim(top=ymax * 1.15)   # 15% for slowdowns
    ax.set_xticks(idx)
    ax.set_xticklabels([f"{int(v)}" for v in x_axis_values])
    ax.set_xlabel("Vector size")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Plain vs FHE end-to-end timing (stacked phases)")
    #ax.set_yscale("log")
    ax.legend(ncols=2, fontsize=9, loc="lower right")
    ax.grid(True, which="both", axis="y", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()
    

def init_he(SCHEME, POLY_MOD_DEGREE, SCALE, QI_SIZES):
    """Pyfhel init."""
    HE = Pyfhel()
    HE.contextGen(
        scheme=SCHEME,
        n=POLY_MOD_DEGREE,
        scale=SCALE,
        qi_sizes=QI_SIZES,
    )
    HE.keyGen()
    HE.relinKeyGen()
    HE.rotateKeyGen()
    return HE


def benchmark_linear_regression(
    HE: Pyfhel,
    vector_size: int,
    n_runs_plain: int = 10,
):
    """
    Benchmark a single linear regression inference using CKKS:
        y = w · x + b

    - x: encrypted feature vector of length `vector_size`
    - w, b: model parameters kept in plaintext

    Metrics:
      - plain_time_ms:   average time of plaintext inference (no HE)
      - encrypt_time_ms: time to encrypt x
      - compute_time_ms: time for homomorphic dot product and bias add
      - decrypt_time_ms: time to decrypt the final result
      - mae:             |y_plain - y_he| (CKKS is approximate)
    """
    print(f"\n===== CKKS Linear Regression Benchmark (VECTOR_SIZE={vector_size}) =====")

    # --- Check that vector fits into available CKKS slots ---
    try:
        n_slots = HE.get_nSlots()
    except AttributeError:
        # For CKKS, slots ≈ n/2 if not available explicitly
        # (but in modern Pyfhel get_nSlots() should exist)
        n_slots = None

    if n_slots is not None and vector_size > n_slots:
        raise ValueError(
            f"VECTOR_SIZE={vector_size} > n_slots={n_slots}. "
            f"Reduce VECTOR_SIZE or increase poly_mod_degree."
        )

    # -------- 1. Generate synthetic data for linear regression --------
    # Use floats to leverage CKKS approximate arithmetic.
    x = np.random.rand(vector_size)  # feature vector
    w = np.random.rand(vector_size)  # weight vector
    b = float(np.random.rand())  # scalar bias

    # -------- 2. Plain (non-HE) inference benchmark --------
    t0 = time.perf_counter()
    for _ in range(n_runs_plain):
        y_plain = float(w @ x + b)
    t1 = time.perf_counter()
    plain_time_ms = (t1 - t0) * 1000.0 / n_runs_plain

    # -------- 3. Encrypt input vector x --------
    t_enc0 = time.perf_counter()
    ctxt_x = HE.encryptFrac(x)  # x -> ciphertext (vector packed into slots)
    t_enc1 = time.perf_counter()
    encrypt_time_ms = (t_enc1 - t_enc0) * 1000.0

    # -------- 4. Prepare plaintext weight vector and bias --------
    # Weights can be encoded before the work, not timer is needed.
    ptxt_w = HE.encodeFrac(w)

    # Bias is encoded as a single-slot plaintext vector [b].
    # Only slot 0 will matter after the dot-product reduction.
    ptxt_b = HE.encodeFrac(np.array([b], dtype=float))

    # -------- 5. Homomorphic computation: y = w · x + b --------
    t_comp0 = time.perf_counter()
    ctxt_dot = ctxt_x @ ptxt_w  # homomorphic dot product
    ctxt_y = ctxt_dot + ptxt_b  # add bias (ciphertext + ciphertext)

    t_comp1 = time.perf_counter()
    compute_time_ms = (t_comp1 - t_comp0) * 1000.0

    # -------- 6. Decrypt result --------
    t_dec0 = time.perf_counter()
    dec_vec = HE.decryptFrac(ctxt_y)
    t_dec1 = time.perf_counter()
    decrypt_time_ms = (t_dec1 - t_dec0) * 1000.0

    # After reduction, the dot-product + bias is in the first slot.
    y_he = float(dec_vec[0])

    # -------- 7. Error metric (CKKS is approximate) --------
    mae = abs(y_plain - y_he)
    nmae = mae / y_plain
    slow_down = (decrypt_time_ms + encrypt_time_ms + compute_time_ms)/plain_time_ms
    # -------- 8. Print and return results --------
    #print(
    #    f"Plain (no HE) time: {plain_time_ms:.3f} ms "
    #    f"(avg over {n_runs_plain} runs)"
    #)
    print(f"Plain (no HE) time (ms): {plain_time_ms}")
    print(f"Encrypt time (ms): {encrypt_time_ms}")
    print(f"HE compute time (ms): {compute_time_ms}")
    print(f"Decrypt time (ms): {decrypt_time_ms}")
    print(f"Plain result: {y_plain:.6f}")
    print(f"HE result (slot 0): {y_he:.6f}")
    print(f"Absolute error (MAE): {mae:.3e}")
    print(f"Normilized (MAE): {nmae:.3e}")
    print(f"Slow-down: {nmae:.3e}")

    return {
        "vector_size": vector_size,
        "plain_time_ms": plain_time_ms,
        "encrypt_time_ms": encrypt_time_ms,
        "compute_time_ms": compute_time_ms,
        "decrypt_time_ms": decrypt_time_ms,
        "mae": mae,
        "normilized mae": nmae
    }


def benchmark_vector_ops(HE, vector_size: int):
    print(
        f"\n===== CKKS Vector Operations Benchmark 2x+2y (VECTOR_SIZE={vector_size}) ====="
    )

    # --- random vectors generation ---
    plain1 = np.random.rand(vector_size)
    plain2 = np.random.rand(vector_size)

    enc_start = time.perf_counter()
    # --- sypher ---
    t_start = time.perf_counter()
    ctxt1 = HE.encryptFrac(plain1)
    t_mid = time.perf_counter()
    ctxt2 = HE.encryptFrac(plain2)
    t_end = time.perf_counter()

    enc_time_1 = (t_mid - t_start) * 1000
    enc_time_2 = (t_end - t_mid) * 1000
    enc_total = (t_end - t_start) * 1000

    # --- inference ---
    t_start = time.perf_counter()
    ctxt_res = 3 * ctxt1 + 2 * ctxt2
    t_end = time.perf_counter()

    infer_time = (t_end - t_start) * 1000

    # --- desypher ---
    t_start = time.perf_counter()
    dec_res = HE.decryptFrac(ctxt_res)
    t_end = time.perf_counter()
    dec_time = (t_end - t_start) * 1000
    enc_end = time.perf_counter()

    dec_res = dec_res[:vector_size]

    dec_time = (t_end - t_start) * 1000
    crypted_time = (enc_end - enc_start) * 1000

    # --- Plain time and MAE ---
    t_start = time.perf_counter()
    plain_res = 3 * plain1 + 2 * plain2
    t_end = time.perf_counter()
    plain_time = (t_end - t_start) * 1000
    mae = np.mean(np.abs(plain_res - dec_res))
    nmae = mae / np.mean(np.abs(plain_res))
    slow_down = crypted_time/plain_time

    print(f"plain time (ms): {plain_time}")
    print(f"crypted time (ms): {crypted_time}")

    print(f"encrypt (ms): {enc_total}")
    print(f"inference (3x + 2y)(ms): {infer_time}")
    print(f"decrypt (ms): {dec_time}")

    print(f"MAE: {mae:.3e}")
    print(f"normilized MAE: {nmae:.3e}")
    print(f"slow-down: {slow_down}")

    return {
        "plain_time_ms": plain_time,
        "encrypt_time_ms": enc_total,
        "compute_time_ms": infer_time,
        "decrypt_time_ms": dec_time,
        "mae": mae,
        "nmae": nmae
    }


def init_fc_layer_ckks(
    HE: Pyfhel,
    vector_size: int,
    weight_scale: float | None = None,
):
    """
    Initializes a fully-connected layer:
        input dim  = vector_size
        output dim = vector_size

    Weights W are drawn from a normal distribution (similar to common NN inits):
        W_ij ~ N(0, 1 / sqrt(vector_size))  by default
    Biases b_j ~ N(0, 0.1).

    For this HE experiment we:
      - keep W and b in plaintext for reference (plain forward pass),
      - encrypt each row W[j] and each bias b[j] for HE inference.

    Returns:
      {
        "W":          np.ndarray of shape (vector_size, vector_size),
        "b":          np.ndarray of shape (vector_size,),
        "enc_W_rows": list[PyCtxt] of length vector_size,
        "enc_b":      list[PyCtxt] of length vector_size,
      }
    """
    d = vector_size

    # He-like or Xavier-like scaling: std ~ 1/sqrt(d)
    if weight_scale is None:
        weight_scale = 1.0 / np.sqrt(d)

    # Plain weights and biases (for reference / plain inference)
    W = np.random.normal(loc=0.0, scale=weight_scale, size=(d, d))
    b = np.random.normal(loc=0.0, scale=0.1, size=d)

    enc_W_rows = []
    enc_b = []

    # Encrypt each row of W and each bias element as 1-slot vector.
    for j in range(d):
        row = W[j, :]
        ctxt_row = HE.encrypt(row)  # ciphertext containing row in slots
        enc_W_rows.append(ctxt_row)

        # Encode bias as a single-slot vector [b_j]
        ctxt_bj = HE.encrypt(np.array([b[j]], dtype=float))
        enc_b.append(ctxt_bj)

    return {
        "W": W,
        "b": b,
        "enc_W_rows": enc_W_rows,
        "enc_b": enc_b,
    }


def activation_square(z: np.ndarray) -> np.ndarray:
    """
    Simple polynomial activation: element-wise square.
    This is HE-friendly (no comparisons, only multiplications).
    """
    return z * z


def benchmark_fc_layer_ckks(
    HE: Pyfhel,
    layer: dict,
    vector_size: int,
    n_tests: int = 5,
    use_activation: bool = True,
):
    """
    Benchmarks a single fully-connected layer:
        y = W @ x + b
    with optional activation:
        y = activation(y)

    - Input dimension:  vector_size
    - Output dimension: vector_size

    The layer dict is expected to come from init_fc_layer_ckks and contain:
      - "W":          np.ndarray (d, d)
      - "b":          np.ndarray (d,)
      - "enc_W_rows": list of encrypted rows W[j]
      - "enc_b":      list of encrypted biases b[j] as 1-slot vectors

    HE pipeline per test:
      1) encrypt input vector x
      2) for each output neuron j:
           ctxt_dot_j = ctxt_x @ enc_W_rows[j]    (homomorphic dot product)
           ctxt_yj    = ctxt_dot_j + enc_b[j]     (add bias)
           optional:   ctxt_yj = ctxt_yj * ctxt_yj  (square activation)
           decrypt ctxt_yj and take slot 0
      3) compare HE outputs against plain outputs (MAE over the whole vector)

    Returns:
      metrics dict with average timings and errors.
    """
    d = vector_size
    W = layer["W"]
    b = layer["b"]
    enc_W_rows = layer["enc_W_rows"]
    enc_b = layer["enc_b"]

    assert len(enc_W_rows) == d
    assert len(enc_b) == d

    encrypt_times = []
    compute_times = []
    decrypt_times = []
    mae_list = []
    nmae_list = []
    plain_times = []

    for test_idx in range(n_tests):
        # -------- 1. Generate random input --------
        x = np.random.rand(d)

        # -------- 2. Plain forward pass --------
        t_enc0 = time.perf_counter()
        y_plain = W @ x + b
        if use_activation:
            y_plain = activation_square(y_plain)
        t_enc1 = time.perf_counter()
        plain_times.append((t_enc1 - t_enc0) * 1000.0)

        # -------- 3. HE: encrypt input --------
        t_enc0 = time.perf_counter()
        ctxt_x = HE.encrypt(x)
        t_enc1 = time.perf_counter()
        encrypt_times.append((t_enc1 - t_enc0) * 1000.0)

        # -------- 4. HE: fully-connected layer + activation --------
        

        y_he = np.zeros(d, dtype=float)
        t_compute = 0
        t_decrypt = 0
        for j in range(d):
            t_comp0 = time.perf_counter()
            # Homomorphic dot product: <W[j, :], x>
            ctxt_dot_j = ctxt_x @ enc_W_rows[j]

            # Add bias for neuron j (ciphertext + ciphertext)
            ctxt_yj = ctxt_dot_j + enc_b[j]

            # Optional activation: square
            if use_activation:
                ctxt_yj = ctxt_yj * ctxt_yj
            t_comp1 = time.perf_counter()
            t_compute += (t_comp1 - t_comp0)
            
            
            # Decrypt and take the first slot as the output value
            t_comp0 = time.perf_counter()
            dec_vec_j = HE.decrypt(ctxt_yj)
            t_comp1 = time.perf_counter()
            t_decrypt += (t_comp1 - t_comp0) 
            y_he[j] = float(dec_vec_j[0])

        compute_times.append((t_compute) * 1000.0)
        decrypt_times.append((t_decrypt) * 1000.0)

        # -------- 5. Error over the whole output vector --------
        mae = float(np.mean(np.abs(y_plain - y_he)))
        nmae = mae / np.abs(np.mean(y_plain))
        mae_list.append(mae)
        nmae_list.append(nmae)

        print(f"[Test {test_idx+1}/{n_tests}] MAE = {mae:.3e}")

    metrics = {
        "vector_size": d,
        "n_tests": n_tests,
        "encrypt_time_ms": float(np.mean(encrypt_times)),
        "compute_time_ms": float(np.mean(compute_times)),
        "decrypt_time_ms": float(np.mean(decrypt_times)),
        "mae_avg": float(np.mean(mae_list)),
        "mae_max": float(np.max(mae_list)),
        "nmae_avg": float(np.mean(nmae_list)),
        "nmae_max": float(np.max(nmae_list)),
        "plain_time_ms": float(np.mean(plain_times))
    }
    #slow_down = metrics['plain_time_ms'] / (metrics['compute_time_ms'] + metrics['encrypt_time_ms_avg'])
    print("\n==== FC layer CKKS benchmark summary ====")
    print(f"Vector size: {d}")
    print(f"Tests: {n_tests}")
    print(f"Avg plain time (ms): {metrics['plain_time_ms']:.3f}")
    print(f"Avg encrypt time (ms): {metrics['encrypt_time_ms']:.3f}")
    print(f"Avg HE compute (ms):{metrics['compute_time_ms']:.3f}")
    print(f"Avg decrypt (ms):{metrics['decrypt_time_ms']:.3f}")
    print(f"Avg MAE: {metrics['mae_avg']:.3e}")
    print(f"Max MAE: {metrics['mae_max']:.3e}")
    print(f"Avg normilized MAE: {metrics['nmae_avg']:.3e}")
    print(f"Max normilized MAE: {metrics['nmae_max']:.3e}")
    #print(f"Slow-down: {slow_down}")

    return metrics


def main():
    # ====== Hyper parameters ======
    VECTOR_SIZE = 10000
    POLY_MOD_DEGREE = 2**15  # n (must be >= 2*VECTOR_SIZE)
    SCALE = 2**30
    QI_SIZES = [60, 30, 30, 60]
    SCHEME = "CKKS"
    
    
    
    # size:
    #for VECTOR_SIZE in [10, 100, 1000, 10000]:
        #HE = init_he(SCHEME, POLY_MOD_DEGREE, SCALE, QI_SIZES)
        #benchmark_linear_regression(HE, VECTOR_SIZE)
        #benchmark_vector_ops(HE, VECTOR_SIZE)

    VECTOR_SIZE = 100
    for degree in [13, 14, 15]:
        POLY_MOD_DEGREE = 2**degree
        HE = init_he(SCHEME, POLY_MOD_DEGREE, SCALE, QI_SIZES)
        #benchmark_linear_regression(HE, VECTOR_SIZE)
        #benchmark_vector_ops(HE, VECTOR_SIZE)
    

    # for size in [16, 64, 256, 1024, 4096, 8192]:
    #     benchmark_vector_ops(HE, size)


    for VECTOR_SIZE in [10, 20, 30]:
        HE = init_he(SCHEME, POLY_MOD_DEGREE, SCALE, QI_SIZES)
        layer = init_fc_layer_ckks(HE, vector_size=VECTOR_SIZE)
        benchmark_fc_layer_ckks(
            HE,
            layer,
            vector_size=VECTOR_SIZE,
            n_tests=3,
            use_activation=True,
        )


if __name__ == "__main__":
    main()
