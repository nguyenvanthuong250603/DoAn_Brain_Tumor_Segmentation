import streamlit as st
import numpy as np
import torch
import cv2
from PIL import Image

# Thư viện cho 3D Visualization
import plotly.graph_objects as go
from skimage import measure

# ============================================================================
# 1. CÁC HÀM XỬ LÝ ẢNH & TIỆN ÍCH
# ============================================================================
SEGMENT_CLASSES = {
    0: "Background",
    1: "Necrotic/Core (Lõi hoại tử)",
    2: "Edema (Phù nề)",
    3: "Enhancing (U bắt thuốc)",
}

CLASS_COLORS = {
    0: (0, 0, 0),  # Đen
    1: (255, 50, 50),  # Đỏ
    2: (50, 255, 50),  # Xanh lá
    3: (50, 50, 255),  # Xanh dương
}
TARGET_SIZE = 240
PIXEL_TO_MM3 = 1.0  # Giả định 1 voxel = 1mm3 (Cần chỉnh nếu có header file gốc)


def zscore_normalization(volume):
    """Chuẩn hóa ảnh để model dễ học"""
    mean = np.mean(volume)
    std = np.std(volume)
    if std < 1e-8:
        return np.zeros_like(volume)
    return (volume - mean) / std


def clean_segmentation_3d(mask_3d):
    """
    🧹 HÀM LỌC RÁC QUAN TRỌNG:
    Chỉ giữ lại khối u liên thông lớn nhất (Largest Connected Component).
    Xóa bỏ các đốm nhiễu nhỏ li ti do model dự đoán sai.
    """
    # Tạo mask nhị phân: Chỗ nào là u (bất kể loại 1,2,3) thì = 1, nền = 0
    binary_mask = mask_3d > 0

    # Tìm các khối liên thông trong không gian 3D
    labels = measure.label(binary_mask)

    # Nếu không tìm thấy khối u nào
    if labels.max() == 0:
        return mask_3d

    # Tính thể tích từng khối (đếm số pixel của từng label)
    regions = measure.regionprops(labels)

    # Tìm khối có diện tích lớn nhất
    largest_region = max(regions, key=lambda r: r.area)

    # Tạo mask sạch: Chỉ giữ lại vị trí của khối lớn nhất
    # Nhân với mask gốc để phục hồi lại các nhãn 1, 2, 3
    cleaned_mask = mask_3d * (labels == largest_region.label)

    return cleaned_mask


def calculate_tumor_volume_slice(pred_mask):
    unique, counts = np.unique(pred_mask, return_counts=True)
    stats = dict(zip(unique, counts))
    return {
        "NCR": stats.get(1, 0) * PIXEL_TO_MM3,
        "ED": stats.get(2, 0) * PIXEL_TO_MM3,
        "ET": stats.get(3, 0) * PIXEL_TO_MM3,
        "TOTAL": (stats.get(1, 0) + stats.get(2, 0) + stats.get(3, 0)) * PIXEL_TO_MM3,
    }


def create_color_mask(pred_mask):
    h, w = pred_mask.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, color in CLASS_COLORS.items():
        if cid == 0:
            continue
        color_img[pred_mask == cid] = color
    return color_img


def create_overlay(bg_img, mask_img, alpha=0.4):
    bg_norm = cv2.normalize(bg_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    bg_rgb = cv2.cvtColor(bg_norm, cv2.COLOR_GRAY2RGB)
    if bg_rgb.shape[:2] != mask_img.shape[:2]:
        mask_img = cv2.resize(
            mask_img,
            (bg_rgb.shape[1], bg_rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    return Image.fromarray(cv2.addWeighted(bg_rgb, 1 - alpha, mask_img, alpha, 0))


# ============================================================================
# 2. HÀM DỰ ĐOÁN & DỰNG HÌNH 3D
# ============================================================================
def predict_whole_volume(model, device, vol_data, batch_size=16):
    """
    Dự đoán toàn bộ volume 3D, có áp dụng lọc nhiễu.
    """
    depth = vol_data["flair"].shape[-1]

    # Khởi tạo mảng 3D
    full_mask_3d = np.zeros((TARGET_SIZE, TARGET_SIZE, depth), dtype=np.uint8)
    full_brain_3d = np.zeros((TARGET_SIZE, TARGET_SIZE, depth), dtype=np.float32)

    progress_bar = st.progress(0)
    status_text = st.empty()
    model.eval()

    # --- BƯỚC 1: DỰ ĐOÁN TỪNG BATCH ---
    for i in range(0, depth, batch_size):
        end = min(i + batch_size, depth)
        batch_frames = []

        # Chuẩn bị batch
        for idx in range(i, end):
            # Lấy slice và Resize
            # ⚠️ QUAN TRỌNG: Cần chuẩn hóa Z-Score trước khi đưa vào model
            s_flair = zscore_normalization(
                cv2.resize(vol_data["flair"][:, :, idx], (TARGET_SIZE, TARGET_SIZE))
            )
            s_t1 = zscore_normalization(
                cv2.resize(vol_data["t1"][:, :, idx], (TARGET_SIZE, TARGET_SIZE))
            )
            s_t1ce = zscore_normalization(
                cv2.resize(vol_data["t1ce"][:, :, idx], (TARGET_SIZE, TARGET_SIZE))
            )
            s_t2 = zscore_normalization(
                cv2.resize(vol_data["t2"][:, :, idx], (TARGET_SIZE, TARGET_SIZE))
            )

            # Lưu lại T1 slice (chưa chuẩn hóa hoặc chuẩn hóa nhẹ để hiển thị vỏ não)
            # Ở đây ta lưu bản gốc resize để vẽ vỏ não cho đẹp
            t1_original = cv2.resize(
                vol_data["t1"][:, :, idx], (TARGET_SIZE, TARGET_SIZE)
            )
            full_brain_3d[:, :, idx] = t1_original

            stack = np.stack([s_flair, s_t1, s_t1ce, s_t2], axis=0).astype(np.float32)
            batch_frames.append(stack)

        if not batch_frames:
            continue

        batch_tensor = torch.from_numpy(np.array(batch_frames)).to(device)
        with torch.no_grad():
            output = model(batch_tensor)
            preds = torch.argmax(output, dim=1).cpu().numpy()  # (Batch, H, W)

        # Lưu kết quả vào mảng 3D tạm thời
        for k, p in enumerate(preds):
            slice_idx = i + k
            full_mask_3d[:, :, slice_idx] = p

        progress_bar.progress(min(end / depth, 1.0))
        status_text.text(f"Đang dự đoán layer: {end}/{depth}")

    # --- BƯỚC 2: HẬU XỬ LÝ (LỌC NHIỄU) ---
    status_text.text("Đang lọc nhiễu và tính toán thể tích...")

    # Gọi hàm lọc rác để xóa các khối u rời rạc
    clean_mask_3d = clean_segmentation_3d(full_mask_3d)

    # --- BƯỚC 3: TÍNH TOÁN THỂ TÍCH TRÊN MASK ĐÃ LỌC ---
    total_mm3 = {"NCR": 0.0, "ED": 0.0, "ET": 0.0, "TOTAL": 0.0}

    # Duyệt qua toàn bộ volume 3D đã làm sạch để cộng dồn thể tích
    unique, counts = np.unique(clean_mask_3d, return_counts=True)
    stats_all = dict(zip(unique, counts))

    total_mm3["NCR"] = stats_all.get(1, 0) * PIXEL_TO_MM3
    total_mm3["ED"] = stats_all.get(2, 0) * PIXEL_TO_MM3
    total_mm3["ET"] = stats_all.get(3, 0) * PIXEL_TO_MM3
    total_mm3["TOTAL"] = sum(total_mm3.values())

    status_text.empty()
    progress_bar.empty()

    final_stats = {k: v / 1000.0 for k, v in total_mm3.items()}  # Đổi sang cm3

    # Trả về mask đã làm sạch
    return final_stats, clean_mask_3d, full_brain_3d


def plot_3d_tumor(volume_mask, brain_volume):
    """
    Vẽ não bộ trong suốt và khối u bên trong
    """
    fig = go.Figure()

    # 1. VẼ VỎ NÃO (Dùng dữ liệu T1)
    try:
        brain_norm = (brain_volume - brain_volume.min()) / (
            brain_volume.max() - brain_volume.min()
        )

        # step_size=2: Giúp render nhanh hơn và làm mượt bề mặt
        verts_b, faces_b, _, _ = measure.marching_cubes(
            brain_norm, level=0.15, step_size=2
        )

        fig.add_trace(
            go.Mesh3d(
                x=verts_b[:, 0],
                y=verts_b[:, 1],
                z=verts_b[:, 2],
                i=faces_b[:, 0],
                j=faces_b[:, 1],
                k=faces_b[:, 2],
                opacity=0.08,  # Rất trong suốt
                color="lightgray",
                name="Cấu trúc Não",
                showlegend=True,
                hoverinfo="skip",
            )
        )
    except Exception as e:
        print(f"Không thể vẽ vỏ não (có thể do nền đen quá nhiều): {e}")

    # 2. VẼ KHỐI U (Dùng mask dự đoán)
    classes = [
        (1, "Lõi Hoại tử (NCR)", "red", 1.0),
        (3, "U Bắt thuốc (ET)", "blue", 0.5),
        (2, "Phù nề (ED)", "green", 0.15),
    ]

    has_tumor = False
    for class_id, name, color, opacity in classes:
        mask = volume_mask == class_id
        if not np.any(mask):
            continue

        try:
            # step_size=1 hoặc 2 để khối u chi tiết hơn vỏ não
            verts, faces, _, _ = measure.marching_cubes(mask, level=0.5, step_size=1)

            fig.add_trace(
                go.Mesh3d(
                    x=verts[:, 0],
                    y=verts[:, 1],
                    z=verts[:, 2],
                    i=faces[:, 0],
                    j=faces[:, 1],
                    k=faces[:, 2],
                    opacity=opacity,
                    color=color,
                    name=name,
                    showlegend=True,
                )
            )
            has_tumor = True
        except Exception:
            continue

    if not has_tumor:
        return None

    # Cấu hình Camera và Ánh sáng
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False, backgroundcolor="black"),
            yaxis=dict(visible=False, backgroundcolor="black"),
            zaxis=dict(visible=False, backgroundcolor="black"),
            aspectmode="data",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
        ),
        title="Mô hình Não 3D & Khối u (Đã lọc nhiễu)",
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(x=0, y=1, font=dict(color="white")),
        paper_bgcolor="black",
    )
    return fig


def norm_show(img):
    """
    Hàm phụ trợ: Chuẩn hóa ảnh về khoảng [0, 255] và ép kiểu sang uint8
    để hiển thị được trên Streamlit/Matplotlib.
    """
    # Tránh lỗi chia cho 0 hoặc ảnh rỗng
    if img is None or img.size == 0:
        return np.zeros((100, 100), dtype=np.uint8)

    # Min-max scaling về 0-255
    img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

    # Ép kiểu thành số nguyên 8-bit (uint8)
    return img_norm.astype(np.uint8)
