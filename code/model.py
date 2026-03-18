import torch
import torch.nn as nn
import google.generativeai as genai
import os
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

MODEL_PATH = r".\models\best_unet_epoch032.pth"


# ===========================================================
# 🧠 MODEL ARCHITECTURE — UNet2D
# ===========================================================
class DoubleConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet2D(nn.Module):
    def __init__(
        self, in_channels=4, out_channels=4, features=(64, 128, 256, 512), dropout=0.1
    ):
        super().__init__()
        f = features
        self.inc = DoubleConv2d(in_channels, f[0], dropout=0)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv2d(f[0], f[1], dropout=dropout)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv2d(f[1], f[2], dropout=dropout)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv2d(f[2], f[3], dropout=dropout)
        )
        self.up3 = nn.ConvTranspose2d(f[3], f[2], 2, stride=2)
        self.dec3 = DoubleConv2d(f[3], f[2], dropout=dropout)
        self.up2 = nn.ConvTranspose2d(f[2], f[1], 2, stride=2)
        self.dec2 = DoubleConv2d(f[2], f[1], dropout=dropout)
        self.up1 = nn.ConvTranspose2d(f[1], f[0], 2, stride=2)
        self.dec1 = DoubleConv2d(f[1], f[0], dropout=0)
        self.outc = nn.Conv2d(f[0], out_channels, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        e0 = self.inc(x)
        e1 = self.down1(e0)
        e2 = self.down2(e1)
        e3 = self.down3(e2)
        d3 = self.dec3(torch.cat([self.up3(e3), e2], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e1], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e0], 1))
        return self.outc(d1)


# ===========================================================
# 🛠️ HELPER FUNCTION TO LOAD MODEL
# ===========================================================
def load_pretrained_model(model_path, device, in_channels=2, out_channels=4):

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Không tìm thấy file model tại: {model_path}")

    # Khởi tạo kiến trúc model
    model = UNet2D(in_channels=in_channels, out_channels=out_channels).to(device)

    # --- SỬA LỖI Ở ĐÂY ---
    # Thêm weights_only=False để cho phép load numpy scalars
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Xử lý các trường hợp lưu checkpoint khác nhau
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()
    return model


def create_genai_client(api_key: str = None):
    """Tạo và trả về instance model Gemini."""
    if api_key is None:
        api_key = os.getenv("GENAI_API_KEY")

    genai.configure(api_key=api_key)

    model = genai.GenerativeModel("gemini-2.5-flash-lite")
    return model


def get_ai_diagnosis(stats_slice_mm3, stats_3d_cm3=None, slice_idx=None):
    try:
        model = create_genai_client()
        data_context = f"""
        1. PHÂN TÍCH LÁT CẮT (Slice {slice_idx}):
        - Hoại tử: {stats_slice_mm3['NCR']:.1f} mm³
        - Phù nề: {stats_slice_mm3['ED']:.1f} mm³
        - Bắt thuốc: {stats_slice_mm3['ET']:.1f} mm³
        """
        if stats_3d_cm3:
            data_context += f"""
            2. TỔNG QUAN 3D (Toàn khối u):
            - Tổng Hoại tử: {stats_3d_cm3['NCR']:.2f} cm³
            - Tổng Phù nề: {stats_3d_cm3['ED']:.2f} cm³
            - Tổng Bắt thuốc: {stats_3d_cm3['ET']:.2f} cm³
            - TỔNG THỂ TÍCH: {stats_3d_cm3['TOTAL']:.2f} cm³
            """
        prompt = f"""
        Đóng vai Bác sĩ Chẩn đoán hình ảnh thần kinh. Dựa trên số liệu MRI:
        {data_context}
        YÊU CẦU:
        1. Nhận định hình thái và cấu trúc u.
        2. Dự báo độ ác tính dựa trên thể tích 3D.
        3. Khuyến nghị điều trị.
        Trả lời tiếng Việt, chuyên môn.
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Lỗi AI: {str(e)}"


# ============================================================================
# 3. LOAD MODEL
# ============================================================================
@st.cache_resource
def get_model():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_pretrained_model(MODEL_PATH, device, in_channels=4, out_channels=4)
        return model, device
    except Exception as e:
        return None, str(e)


def test_print():
    print("Model loaded successfully!")
