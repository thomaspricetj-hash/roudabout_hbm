import time
import numpy as np
import zstandard as zstd
import lz4.frame as lz4
import bitdrop_v2  # your Python extension module

# ------------------------------------------------------------
# Generate the perfect BitDrop3D showcase file (only once)
# ------------------------------------------------------------
def generate_showcase_file(path="bitdrop_showcase.bin"):
    N = 256
    x = np.linspace(0, 20, N)
    y = np.linspace(0, 20, N)
    z = np.linspace(0, 20, N)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    field = (
        np.sin(X * 1.3) * 1200 +
        np.cos(Y * 0.7) * 900 +
        np.sin(Z * 1.1) * 1500 +
        np.cos((X+Y+Z) * 0.2) * 500
    ).astype(np.int16)

    field.tofile(path)
    return path

# ------------------------------------------------------------
# Benchmark helper
# ------------------------------------------------------------
def bench(name, compress_fn, decompress_fn, data):
    start = time.time()
    comp = compress_fn(data)
    enc_ms = (time.time() - start) * 1000

    start = time.time()
    dec = decompress_fn(comp)
    dec_ms = (time.time() - start) * 1000

    assert dec == data, f"{name} failed roundtrip"

    ratio = len(comp) / len(data)
    mb = len(data) / (1024 * 1024)
    enc_speed = mb / (enc_ms / 1000)
    dec_speed = mb / (dec_ms / 1000)

    return {
        "name": name,
        "compressed_size": len(comp),
        "ratio": ratio,
        "encode_ms": enc_ms,
        "decode_ms": dec_ms,
        "encode_mb_s": enc_speed,
        "decode_mb_s": dec_speed,
    }

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    path = generate_showcase_file()
    data = open(path, "rb").read()

    print("Running benchmarks on:", path)
    print("Original size:", len(data), "bytes")

    # BitDrop3D
    bd = bench(
        "BitDrop3D",
        lambda d: bitdrop_v2.compress(d),
        lambda d: bitdrop_v2.decompress(d),
        data
    )

    # zstd
    cctx = zstd.ZstdCompressor(level=5)
    dctx = zstd.ZstdDecompressor()

    z = bench(
        "zstd",
        lambda d: cctx.compress(d),
        lambda d: dctx.decompress(d),
        data
    )

    # lz4
    l = bench(
        "lz4",
        lambda d: lz4.compress(d),
        lambda d: lz4.decompress(d),
        data
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------
    print("\n=== Benchmark Results ===")
    for r in [bd, z, l]:
        print(f"\n{r['name']}")
        print(f"  Compressed size: {r['compressed_size']}")
        print(f"  Ratio:           {r['ratio']:.6f}")
        print(f"  Encode:          {r['encode_ms']:.3f} ms ({r['encode_mb_s']:.1f} MB/s)")
        print(f"  Decode:          {r['decode_ms']:.3f} ms ({r['decode_mb_s']:.1f} MB/s)")
