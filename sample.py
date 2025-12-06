import os
import sys
import math

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Option 1：DCT
# =========================

def psnr(a, b):
    # PSNR，測量重建影像品質（我寫報告的時候方便比較）。
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return 100.0
    return 10 * math.log10((255.0 ** 2) / mse)


def dct_display_image(dct_img):
    # 以 log 縮放視覺化頻譜，讓很小的係數也看得到。
    mag = np.log(np.abs(dct_img) + 1.0)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return mag.astype(np.uint8)


def Global_DCT(img):
    # 對整張影像做 2D DCT/IDCT。
    # TODO:
    #   1. Use OpenCV’s DCT / numpy fft or implement per formula.
    #      e.g., cv.dct(img_float32) <--> cv.idct(dct_img).
    #   2. Return both DCT result and reconstructed image (optional).
    h, w = img.shape
    pad_h = h % 2
    pad_w = w % 2
    padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, borderType=cv2.BORDER_REFLECT)
    img_f = padded.astype(np.float32)
    dct_img = cv2.dct(img_f)
    rec = cv2.idct(dct_img)
    rec = rec[:h, :w]
    rec = np.clip(rec, 0, 255).astype(np.uint8)
    return dct_img, rec


def Fast_DCT_separable(img):
    # 可分離 1D：先對每一列做 1D DCT，再對每一欄做 1D DCT（fast DCT 的做法）。
    # 講義的 Row-Column Decomposition。
    h, w = img.shape
    pad_h = h % 2
    pad_w = w % 2
    padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, borderType=cv2.BORDER_REFLECT)
    f = padded.astype(np.float32)

    # 每一列做 1D DCT
    row_dct = np.zeros_like(f, dtype=np.float32)
    for y in range(f.shape[0]):
        row_dct[y, :] = cv2.dct(f[y, :].reshape(1, -1))[0]

    # 每一欄做 1D DCT
    col_dct = np.zeros_like(row_dct, dtype=np.float32)
    for x in range(row_dct.shape[1]):
        col_dct[:, x] = cv2.dct(row_dct[:, x].reshape(1, -1))[0]

    # 做反轉：先每欄 1D IDCT，再每列 1D IDCT
    col_idct = np.zeros_like(col_dct, dtype=np.float32)
    for x in range(col_dct.shape[1]):
        col_idct[:, x] = cv2.idct(col_dct[:, x].reshape(1, -1))[0]

    rec_full = np.zeros_like(col_idct, dtype=np.float32)
    for y in range(col_idct.shape[0]):
        rec_full[y, :] = cv2.idct(col_idct[y, :].reshape(1, -1))[0]

    rec = rec_full[:h, :w]
    rec = np.clip(rec, 0, 255).astype(np.uint8)
    return col_dct, rec


def Local_DCT(img, kernel_size=8):
    # 區塊式 DCT/IDCT，並補邊讓尺寸可以被區塊整除。
    # TODO:
    #   1. Divide the image into non-overlapping blocks (kernel_size x kernel_size).
    #   2. Apply 2D DCT to each block (OpenCV / numpy / formula).
    #   3. Reconstruct image from block-wise IDCT (optional).
    h, w = img.shape
    pad_h = (kernel_size - h % kernel_size) % kernel_size
    pad_w = (kernel_size - w % kernel_size) % kernel_size
    padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, borderType=cv2.BORDER_REFLECT)

    dct_padded = np.zeros_like(padded, dtype=np.float32)
    rec_padded = np.zeros_like(padded, dtype=np.float32)

    for y in range(0, padded.shape[0], kernel_size):
        for x in range(0, padded.shape[1], kernel_size):
            block = padded[y:y + kernel_size, x:x + kernel_size].astype(np.float32)
            dct_block = cv2.dct(block)
            dct_padded[y:y + kernel_size, x:x + kernel_size] = dct_block
            rec_block = cv2.idct(dct_block)
            rec_padded[y:y + kernel_size, x:x + kernel_size] = rec_block

    rec = rec_padded[:h, :w]
    rec = np.clip(rec, 0, 255).astype(np.uint8)
    return dct_padded, rec


def frequency_Domain_filter(DCT_img, keep_size, orig_shape=None):
    # 只保留左上 keep_size x keep_size 的低頻係數，其它地方清零，再做 IDCT。
    # TODO:
    #   1. Input the DCT frequency-domain image.
    #   2. Crop the low-frequency area (upper-left corner, size is arbitrary).
    #   3. Set other coefficients to zero (or keep as is).
    #   4. Apply IDCT to reconstruct the filtered image.
    #   5. Return the filtered DCT and reconstructed image.
    k = int(keep_size)
    h, w = DCT_img.shape
    k = max(1, min(k, h, w))
    filtered = np.zeros_like(DCT_img, dtype=np.float32)
    filtered[:k, :k] = DCT_img[:k, :k]
    rec = cv2.idct(filtered)
    if orig_shape is not None:
        rec = rec[:orig_shape[0], :orig_shape[1]]
    rec = np.clip(rec, 0, 255).astype(np.uint8)
    return filtered, rec


# =========================
# Option 2: Vector Quantization (VQ)
# =========================

def extract_blocks(img, block_size=4):
    # TODO:
    #   1. Divide the image into non-overlapping blocks (block_size x block_size).
    #   2. Flatten each block to vectors and collect them.
    #   3. Return vectors and shape info for reconstruction.
    h, w = img.shape
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size
    padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, borderType=cv2.BORDER_REFLECT)
    blocks = []
    for y in range(0, padded.shape[0], block_size):
        for x in range(0, padded.shape[1], block_size):
            blk = padded[y:y + block_size, x:x + block_size].astype(np.float32)
            blocks.append(blk.flatten())
    vectors = np.vstack(blocks)
    return vectors, (h, w), padded.shape


def lbg_codebook_training(vectors, codebook_size=64, epsilon=1e-3, max_iter=100):
    # TODO (Linde–Buzo–Gray, lecture Ch7 p.46):
    #   1. Initialize codebook (e.g., random vectors).
    #   2. Iterate until convergence:
    #       a. Assign each vector to nearest codeword (clustering).
    #       b. Update codeword as mean of assigned vectors.
    #       c. Compute distortion and check stop condition.
    #   3. Return final codebook.
    np.random.seed(0)
    n, dim = vectors.shape
    if n == 0:
        raise ValueError("No vectors to train on.")

    # [p. 46] 初始化：從訓練向量中隨機選取 N_c 個向量作為初始 Codebook。
    choice = np.random.choice(n, size=codebook_size, replace=n < codebook_size)
    codebook = vectors[choice].copy()

    prev_distortion = None
    for _ in range(max_iter):
        # [p. 46] 分群
        # a. Assign
        dists = np.sum((vectors[:, None, :] - codebook[None, :, :]) ** 2, axis=2)
        labels = np.argmin(dists, axis=1)
        min_dists = dists[np.arange(n), labels]
        distortion = np.mean(min_dists)

        # [p. 46] 更新 Codebook
        # b. Update
        # 計算每個群集的重心作為新的 codeword。
        new_codebook = np.zeros_like(codebook)
        for k in range(codebook_size):
            mask = labels == k
            if np.any(mask):
                new_codebook[k] = vectors[mask].mean(axis=0)
            else:
                # 如果某個群集沒有分到任何向量，就隨機抽一個向量填回該 codeword，去避免空群。
                new_codebook[k] = vectors[np.random.randint(0, n)]

        codebook = new_codebook

        # c. Check convergence
        if prev_distortion is not None:
            if abs(prev_distortion - distortion) / (prev_distortion + 1e-9) < epsilon:
                break
        prev_distortion = distortion

    return codebook


def vq_encode(vectors, codebook):
    # TODO:
    #   1. For each vector, find nearest codeword.
    #   2. Store the index of nearest codeword.
    #   3. Return index array (VQ encoded representation).
    dists = np.sum((vectors[:, None, :] - codebook[None, :, :]) ** 2, axis=2)
    indices = np.argmin(dists, axis=1)
    return indices


def vq_decode(indices, codebook, image_shape, block_size=4):
    # TODO:
    #   1. Map each index back to its codeword vector.
    #   2. Reshape each codeword to (block_size x block_size).
    #   3. Reconstruct full image using shape info.
    #   4. Return reconstructed image.
    padded_shape, orig_shape = image_shape
    h_pad, w_pad = padded_shape
    h_orig, w_orig = orig_shape
    rec = np.zeros((h_pad, w_pad), dtype=np.float32)
    idx = 0
    for y in range(0, h_pad, block_size):
        for x in range(0, w_pad, block_size):
            vec = codebook[indices[idx]].reshape(block_size, block_size)
            rec[y:y + block_size, x:x + block_size] = vec
            idx += 1
    rec = rec[:h_orig, :w_orig]
    rec = np.clip(rec, 0, 255).astype(np.uint8)
    return rec


def visualize_codebook_as_table(codebook, block_size=4):
    # TODO:
    #   1. Optionally output codebook as table (for report screenshot).
    #   2. Each row is a flattened codeword.
    #   3. Or format as grid image.
    df = pd.DataFrame(codebook)
    return df


if __name__ == '__main__':

    if len(sys.argv) < 2:
        raise SystemExit("Usage: python sample.py <img1> [<img2> ...] [output_dir]")

    args = sys.argv[1:]
    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    if len(args) >= 2 and os.path.splitext(args[-1].lower())[1] not in image_exts:
        out_dir = args[-1]
        img_paths = args[:-1]
    else:
        out_dir = "output_dct"
        img_paths = args

    os.makedirs(out_dir, exist_ok=True)

    for img_path in img_paths:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Skip (cannot read): {img_path}")
            continue

        name = os.path.splitext(os.path.basename(img_path))[0]
        save_dir = os.path.join(out_dir, name)
        os.makedirs(save_dir, exist_ok=True)

        orig_shape = img.shape
        cv2.imwrite(os.path.join(save_dir, "0_original.png"), img)
        # =========================
        # Option 1：DCT
        # =========================
        # 全圖 DCT + 重建
        dct_global, rec_global = Global_DCT(img)
        cv2.imwrite(os.path.join(save_dir, "1_dct_full_vis.png"), dct_display_image(dct_global))
        cv2.imwrite(os.path.join(save_dir, "2_idct_full.png"), rec_global)
        print(f"[{name}] Global DCT PSNR: {psnr(img, rec_global):.2f} dB")

        # 頻域裁切：保留不同尺寸
        keep_sizes = [8, 16, 32, 64]
        for k in keep_sizes:
            filtered, rec = frequency_Domain_filter(dct_global, keep_size=k, orig_shape=orig_shape)
            cv2.imwrite(os.path.join(save_dir, f"3_dct_keep_{k}.png"), dct_display_image(filtered))
            cv2.imwrite(os.path.join(save_dir, f"4_idct_keep_{k}.png"), rec)
            print(f"[{name}] Keep {k}x{k} -> PSNR: {psnr(img, rec):.2f} dB")

        # 區塊式 DCT/IDCT
        dct_local, rec_local = Local_DCT(img)
        cv2.imwrite(os.path.join(save_dir, "5_local_dct_vis.png"), dct_display_image(dct_local))
        cv2.imwrite(os.path.join(save_dir, "6_local_idct.png"), rec_local)
        print(f"[{name}] Local DCT PSNR: {psnr(img, rec_local):.2f} dB")

        # Fast DCT（可分離 1D）
        dct_fast, rec_fast = Fast_DCT_separable(img)
        cv2.imwrite(os.path.join(save_dir, "7_fast_dct_vis.png"), dct_display_image(dct_fast))
        cv2.imwrite(os.path.join(save_dir, "8_fast_idct.png"), rec_fast)
        print(f"[{name}] Fast DCT PSNR: {psnr(img, rec_fast):.2f} dB")

        # =========================
        # Option 2: Vector Quantization (VQ)
        # =========================
        block_size = 4
        codebook_size = 64
        vectors, orig_shape_blocks, padded_shape = extract_blocks(img, block_size=block_size)
        cb = lbg_codebook_training(vectors, codebook_size=codebook_size, epsilon=1e-3, max_iter=50)
        idxs = vq_encode(vectors, cb)
        rec_vq = vq_decode(idxs, cb, image_shape=(padded_shape, orig_shape_blocks), block_size=block_size)
        cv2.imwrite(os.path.join(save_dir, "vq_recon.png"), rec_vq)
        print(f"[{name}] VQ PSNR: {psnr(img, rec_vq):.2f} dB")

        df_cb = visualize_codebook_as_table(cb, block_size=block_size)
        df_cb.to_csv(os.path.join(save_dir, "codebook.csv"), index=False, header=False)

        # 將 codebook 存成表格圖檔案。
        fig, ax = plt.subplots(figsize=(12, min(12, codebook_size * 0.2)))
        ax.axis('off')
        tbl = ax.table(cellText=np.round(cb, 2), loc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(6)
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, "codebook_table.png"), dpi=200)
        plt.close(fig)
