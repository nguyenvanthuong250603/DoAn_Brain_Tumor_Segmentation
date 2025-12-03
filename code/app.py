import streamlit as st
import numpy as np
import h5py
import torch
import cv2
import tempfile
import os
import plotly.graph_objects as go
from skimage import measure
from PIL import Image

# Import các hàm xử lý ảnh mới từ file process_image.py của bạn
# Đảm bảo process_image.py chứa hàm zscore_normalization, clean_segmentation_3d...
from process_image import *

# Import model
try:
    from model import *
except ImportError:
    st.error("❌ Không tìm thấy file 'model_run.py'.")
    st.stop()

st.set_page_config(
    page_title="Brain tumor segmentation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# CSS tối ưu giao diện
st.markdown(
    """
    <style>
        .block-container {padding-top: 1rem; padding-bottom: 2rem;}
        h1 {margin-bottom: 0.5rem;}
        h3 {margin-top: 0.5rem;}
        div.stButton > button {width: 100%;}
        .report-box {padding: 15px; border-radius: 10px; background-color: #f0f2f6; color: black; border: 1px solid #d1d5db;}
        .report-header {font-weight: bold; font-size: 1.1em; margin-bottom: 10px; color: #31333F;}
    </style>
""",
    unsafe_allow_html=True,
)

# ============================================================================
# MAIN APP
# ============================================================================


def main():
    st.title("🧠 Brain Tumor Segmentation")

    model, device = get_model()
    if not model:
        st.error(f"Lỗi load model: {device}")
        st.stop()

    # Session State
    for key in [
        "vol_stats",
        "vol_mask_3d",
        "vol_brain_3d",
        "ai_report",
        "processed_file",
        "last_file",  # Thêm key này để track file
    ]:
        if key not in st.session_state:
            st.session_state[key] = None

    # --- TOP TOOLBAR ---
    with st.expander("📂 Tải file dữ liệu", expanded=True):
        uploaded_file = st.file_uploader(
            "Chọn file MRI (.h5)", type=["h5", "hdf5"], label_visibility="collapsed"
        )

    if uploaded_file:
        # Reset khi upload file mới
        if st.session_state.last_file != uploaded_file.name:
            for key in [
                "vol_stats",
                "vol_mask_3d",
                "vol_brain_3d",
                "ai_report",
                "processed_file",
            ]:
                st.session_state[key] = None
            st.session_state.last_file = uploaded_file.name

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            with h5py.File(tmp_path, "r") as f:
                if not all(k in f.keys() for k in ["flair", "t1", "t1ce", "t2"]):
                    st.error("Thiếu dữ liệu (cần đủ 4 modal: flair, t1, t1ce, t2)")
                    st.stop()

                # ⚠️ THAY ĐỔI QUAN TRỌNG: Load dữ liệu RAW, KHÔNG chuẩn hóa ở đây
                # Vì hàm predict_whole_volume bên process_image đã tự chuẩn hóa rồi.
                raw_vol = {k: f[k][:] for k in ["flair", "t1", "t1ce", "t2"]}
                depth = raw_vol["flair"].shape[-1]

                # --- THANH SLIDER ---
                idx = st.slider(
                    f"🔍 Chọn lát cắt ({depth} slices)", 0, depth - 1, depth // 2
                )

                # --- XỬ LÝ & HIỂN THỊ 2D ---
                # 1. Lấy dữ liệu thô của lát cắt hiện tại
                s_flair_raw = raw_vol["flair"][:, :, idx]
                s_t1_raw = raw_vol["t1"][:, :, idx]
                s_t1ce_raw = raw_vol["t1ce"][:, :, idx]
                s_t2_raw = raw_vol["t2"][:, :, idx]

                # 2. Chuẩn hóa Z-Score THỦ CÔNG cho lát cắt này để đưa vào Model
                # (Vì raw_vol ở trên chưa chuẩn hóa)
                # Lưu ý: preprocess_input chỉ resize và stack, ta cần z-score trước
                inp_flair = zscore_normalization(
                    cv2.resize(s_flair_raw, (TARGET_SIZE, TARGET_SIZE))
                )
                inp_t1 = zscore_normalization(
                    cv2.resize(s_t1_raw, (TARGET_SIZE, TARGET_SIZE))
                )
                inp_t1ce = zscore_normalization(
                    cv2.resize(s_t1ce_raw, (TARGET_SIZE, TARGET_SIZE))
                )
                inp_t2 = zscore_normalization(
                    cv2.resize(s_t2_raw, (TARGET_SIZE, TARGET_SIZE))
                )

                # Stack thành tensor (1, 4, 240, 240)
                inp_stack = np.stack([inp_flair, inp_t1, inp_t1ce, inp_t2], axis=0)
                inp_tensor = torch.from_numpy(inp_stack).unsqueeze(0).float().to(device)

                # Predict 2D
                with torch.no_grad():
                    pred_mask = torch.argmax(model(inp_tensor), dim=1).cpu().numpy()[0]

                stats_slice = calculate_tumor_volume_slice(pred_mask)

                # ================= BỐ CỤC CHÍNH =================
                col_left, col_right = st.columns([1.5, 1])

                # --- CỘT TRÁI: HÌNH ẢNH 2D ---
                with col_left:
                    st.subheader("🖼️ Phân tích Hình ảnh 2D")

                    # Hiển thị 4 ảnh gốc (Dùng hàm norm_show để tự min-max về 0-255 cho đẹp)
                    cols = st.columns(4)
                    for c, img, lbl in zip(
                        cols,
                        [s_flair_raw, s_t1_raw, s_t1ce_raw, s_t2_raw],
                        ["FLAIR", "T1", "T1ce", "T2"],
                    ):
                        c.image(norm_show(img), caption=lbl, use_container_width=True)

                    # Hình kết quả to (Overlay lên ảnh Flair gốc resize)
                    c_res, c_dat = st.columns([1.5, 1])
                    with c_res:
                        overlay = create_overlay(
                            cv2.resize(s_flair_raw, (TARGET_SIZE, TARGET_SIZE)),
                            create_color_mask(pred_mask),
                        )
                        st.image(
                            overlay,
                            caption=f"Phân vùng Slice {idx}",
                            use_container_width=True,
                        )

                    with c_dat:
                        st.caption("📊 **Chỉ số Slice (mm³)**")
                        st.dataframe(
                            [
                                {
                                    "Vùng": "Hoại tử (NCR)",
                                    "Giá trị": f"{stats_slice['NCR']:.1f}",
                                },
                                {
                                    "Vùng": "Phù nề (ED)",
                                    "Giá trị": f"{stats_slice['ED']:.1f}",
                                },
                                {
                                    "Vùng": "Bắt thuốc (ET)",
                                    "Giá trị": f"{stats_slice['ET']:.1f}",
                                },
                                {
                                    "Vùng": "TỔNG",
                                    "Giá trị": f"{stats_slice['TOTAL']:.1f}",
                                },
                            ],
                            hide_index=True,
                            use_container_width=True,
                        )

                # --- CỘT PHẢI: BÁO CÁO AI ---
                with col_right:
                    st.subheader("🤖 Bác sĩ AI Báo cáo")

                    has_tumor = stats_slice["TOTAL"] > 0
                    if has_tumor:
                        # Kiểm tra xem đã có dữ liệu 3D chưa
                        if st.session_state.processed_file == uploaded_file.name:
                            # Đã có dữ liệu 3D -> Tạo báo cáo
                            current_key = f"{idx}_{stats_slice['TOTAL']}_{st.session_state.vol_stats['TOTAL']}"

                            if (
                                "ai_cache_key" not in st.session_state
                                or st.session_state.ai_cache_key != current_key
                            ):
                                with st.spinner("AI đang soạn báo cáo tổng hợp..."):
                                    diag = get_ai_diagnosis(
                                        stats_slice, st.session_state.vol_stats, idx
                                    )
                                    st.session_state.ai_report = diag
                                    st.session_state.ai_cache_key = current_key

                            with st.container(height=500):
                                if st.session_state.ai_report:
                                    st.markdown(st.session_state.ai_report)
                        else:
                            # Chưa có dữ liệu 3D -> Thông báo đang xử lý
                            st.info(
                                "⏳ Đang phân tích 3D toàn bộ não để có dữ liệu chính xác cho AI..."
                            )
                            st.caption(
                                "Báo cáo sẽ tự động xuất hiện sau khi quá trình quét 3D bên dưới hoàn tất."
                            )
                    else:
                        st.info("Không phát hiện khối u trên lát cắt này.")

                # ================= KHU VỰC DƯỚI: XỬ LÝ & HIỂN THỊ 3D =================
                st.divider()
                st.subheader("🧊 Mô hình 3D & Thể tích Toàn khối")

                # --- LOGIC TỰ ĐỘNG CHẠY 3D ---
                if st.session_state.processed_file != uploaded_file.name:
                    with st.status(
                        "🚀 Đang quét 3D & Lọc nhiễu...", expanded=True
                    ) as status:
                        st.write("Đang tính toán từng lát cắt và dựng hình...")

                        # Gọi hàm predict_whole_volume (Hàm này giờ đã tự normalize và clean rác)
                        s, m, b = predict_whole_volume(model, device, raw_vol)

                        st.session_state.vol_stats = s
                        st.session_state.vol_mask_3d = m
                        st.session_state.vol_brain_3d = b
                        st.session_state.processed_file = uploaded_file.name

                        status.update(
                            label="Hoàn tất xử lý!", state="complete", expanded=False
                        )
                        st.rerun()

                # --- HIỂN THỊ KẾT QUẢ 3D ---
                if st.session_state.vol_stats:
                    c3d_info, c3d_plot = st.columns([1, 3])

                    with c3d_info:
                        v = st.session_state.vol_stats
                        st.success("Dữ liệu 3D (Đã lọc nhiễu)")
                        st.metric(
                            "Tổng thể tích", f"{v['TOTAL']:.2f} cm³", delta="3D Volume"
                        )
                        st.caption(f"🔴 Hoại tử: {v['NCR']:.2f} cm³")
                        st.caption(f"🔵 Bắt thuốc: {v['ET']:.2f} cm³")
                        st.caption(f"🟢 Phù nề: {v['ED']:.2f} cm³")

                    with c3d_plot:
                        if st.session_state.vol_mask_3d is not None and np.any(
                            st.session_state.vol_mask_3d > 0
                        ):
                            fig = plot_3d_tumor(
                                st.session_state.vol_mask_3d,
                                st.session_state.vol_brain_3d,
                            )
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning(
                                "Không phát hiện khối u 3D (Hoặc đã bị lọc bỏ do quá nhỏ)."
                            )

        except Exception as e:
            st.error(f"Lỗi xử lý file: {e}")
            # In chi tiết lỗi ra console để debug nếu cần
            import traceback

            traceback.print_exc()
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


if __name__ == "__main__":
    main()
