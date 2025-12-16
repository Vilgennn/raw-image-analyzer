import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io

# ==========================================
# 核心逻辑函数 (复用之前的算法)
# ==========================================

def detect_shape_from_size(file_bytes_len):
    """
    基于文件大小的自动分辨率推断
    """
    # 常见的传感器分辨率列表 (宽, 高)
    common_resolutions = [
        (2592, 1944), # 5MP
        (1920, 1080), # 1080p
        (1280, 720),  # 720p
        (3264, 2448), # 8MP
        (4000, 3000), # 12MP
        (4608, 3456), # 16MP
        (2048, 2048), (1024, 1024),
        (640, 480)
    ]

    # 检查 8-bit
    for w, h in common_resolutions:
        if file_bytes_len == w * h:
            return w, h, 8
    # 检查 16-bit
    for w, h in common_resolutions:
        if file_bytes_len == w * h * 2:
            return w, h, 16
            
    return None, None, None

def load_raw_data(uploaded_file, width_input, height_input):
    """
    读取 RAW 数据，处理手动输入与自动推断的逻辑
    """
    # 获取文件二进制内容
    bytes_data = uploaded_file.getvalue()
    file_size = len(bytes_data)
    
    final_w, final_h, final_depth = None, None, 8
    
    # 1. 判断使用手动输入还是自动推断
    if width_input > 0 and height_input > 0:
        # === 手动模式 ===
        final_w, final_h = width_input, height_input
        # 简单的位深猜测：如果文件大小是像素数的2倍，则为16位，否则8位
        if file_size >= final_w * final_h * 2:
            final_depth = 16
        else:
            final_depth = 8
        st.info(f"使用手动输入分辨率: {final_w} x {final_h} ({final_depth}-bit)")
    else:
        # === 自动推断模式 ===
        w_auto, h_auto, depth_auto = detect_shape_from_size(file_size)
        if w_auto:
            final_w, final_h, final_depth = w_auto, h_auto, depth_auto
            st.success(f"自动推断分辨率成功: {final_w} x {final_h} ({final_depth}-bit)")
        else:
            st.error(f"无法根据文件大小 ({file_size} bytes) 自动推断分辨率。请在左侧侧边栏手动输入 Width 和 Height。")
            return None

    # 2. 读取数据
    dtype = np.uint16 if final_depth > 8 else np.uint8
    try:
        # 从 bytes 创建 numpy 数组
        data = np.frombuffer(bytes_data, dtype=dtype)
        
        # 截取有效数据 (防止文件比预期大)
        expected_pixels = final_w * final_h
        if len(data) < expected_pixels:
            st.error(f"文件数据不足。需要 {expected_pixels} 像素，实际只有 {len(data)}。")
            return None
            
        data = data[:expected_pixels].reshape((final_h, final_w))
        return data
    except Exception as e:
        st.error(f"解析数据出错: {e}")
        return None

def split_channels_strict(bayer_data):
    """
    严格按照用户指定的映射分离通道
    """
    # G1: row 0, col 0 (根据 Python代码 g1 = bayer_data[0::2, 0::2])
    g1 = bayer_data[0::2, 0::2]
    
    # R: row 0, col 1 (根据 Python代码 r = bayer_data[0::2, 1::2])
    r = bayer_data[0::2, 1::2]
    
    # B: row 1, col 0 (根据 Python代码 b = bayer_data[1::2, 0::2])
    b = bayer_data[1::2, 0::2]
    
    # G2: row 1, col 1 (根据 Python代码 g2 = bayer_data[1::2, 1::2])
    # 注意：用户Prompt中此处可能有笔误写成了G1，但逻辑上必须是G2位置
    g2 = bayer_data[1::2, 1::2]
    
    return {
        'Green Channel 1': g1,
        'Red Channel': r,
        'Blue Channel': b,
        'Green Channel 2': g2
    }

def plot_histogram(data, title, color='gray'):
    """绘制简单的 matplotlib 直方图"""
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(data.ravel(), bins=50, color=color, alpha=0.7)
    ax.set_title(f"{title} Histogram")
    ax.set_xlabel("Pixel Value")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    return fig

# ==========================================
# Streamlit 页面布局
# ==========================================

st.set_page_config(page_title="Raw Bayer Analyzer", layout="wide")

st.title("🔬 Naked Binary RAW 图像分析工具")
st.markdown("""
此工具用于加载无文件头的 `.raw` 图像数据，并分离 Bayer 通道进行分析。
支持 **BRGB** 排列模式 (Odd Rows: B R, Even Rows: G B)。
""")

# --- 侧边栏：输入区域 ---
with st.sidebar:
    st.header("1. 配置与上传")
    
    st.subheader("分辨率设置")
    st.caption("如果不确定，请留空或设为0，系统将尝试根据文件大小自动推断。")
    input_width = st.number_input("Width (px)", min_value=0, value=0, step=1)
    input_height = st.number_input("Height (px)", min_value=0, value=0, step=1)
    
    st.divider()
    
    uploaded_file = st.file_uploader("上传 .raw 文件", type=['raw', 'bin'])

# --- 主区域：处理逻辑 ---

if uploaded_file is not None:
    # 1. 加载数据
    raw_img = load_raw_data(uploaded_file, input_width, input_height)
    
    if raw_img is not None:
        # 2. 分离通道
        channels = split_channels_strict(raw_img)
        
        # 3. 计算统计数据 (Requirement 3: 表格形式)
        stats_list = []
        
        # 添加原始图像统计
        stats_list.append({
            "Channel": "Original RAW",
            "Mean": np.mean(raw_img),
            "Std Dev": np.std(raw_img),
            "Min": np.min(raw_img),
            "Max": np.max(raw_img)
        })
        
        # 添加各通道统计
        for name, data in channels.items():
            stats_list.append({
                "Channel": name,
                "Mean": np.mean(data),
                "Std Dev": np.std(data),
                "Min": np.min(data),
                "Max": np.max(data)
            })
            
        df_stats = pd.DataFrame(stats_list)
        
        # --- 显示区域 ---
        
        st.header("2. 统计信息概览")
        # 将表格设置为高亮显示最大值
        st.dataframe(df_stats.style.highlight_max(axis=0, subset=['Mean', 'Max']), use_container_width=True)
        
        st.divider()
        
        st.header("3. 可视化与直方图")
        
        # Tab 1: 原始图像
        tab_raw, tab_channels = st.tabs(["原始 RAW 图像", "分离通道 (R/B/G1/G2)"])
        
        with tab_raw:
            col1, col2 = st.columns([1, 1])
            with col1:
                st.subheader("原始灰度图")
                # Normalize specifically for display if needed, but st.image handles uint8/uint16 reasonably well if clamp=True
                st.image(raw_img, caption=f"Original RAW ({raw_img.shape[1]}x{raw_img.shape[0]})", clamp=True, channels='GRAY', use_column_width=True)
            with col2:
                st.subheader("灰度直方图")
                fig = plot_histogram(raw_img, "Original RAW", color='black')
                st.pyplot(fig)

        # Tab 2: 分离通道
        with tab_channels:
            # 定义布局：2行2列
            # Row 1
            c1, c2 = st.columns(2)
            
            # Red Channel
            with c1:
                st.markdown("### 🔴 Red Channel")
                data_r = channels['Red Channel']
                st.image(data_r, caption="Red Channel (Odd Row, Odd Col)", clamp=True, channels='GRAY', use_column_width=True)
                st.pyplot(plot_histogram(data_r, "Red Channel", color='red'))
            
            # Blue Channel
            with c2:
                st.markdown("### 🔵 Blue Channel")
                data_b = channels['Blue Channel']
                st.image(data_b, caption="Blue Channel (Even Row, Even Col)", clamp=True, channels='GRAY', use_column_width=True)
                st.pyplot(plot_histogram(data_b, "Blue Channel", color='blue'))
                
            st.divider()
            
            # Row 2
            c3, c4 = st.columns(2)
            
            # Green Channel 1
            with c3:
                st.markdown("### 🟢 Green Channel 1")
                data_g1 = channels['Green Channel 1']
                st.image(data_g1, caption="Green Channel 1 (Odd Row, Even Col)", clamp=True, channels='GRAY', use_column_width=True)
                st.pyplot(plot_histogram(data_g1, "Green Channel 1", color='green'))

            # Green Channel 2
            with c4:
                st.markdown("### 🟢 Green Channel 2")
                data_g2 = channels['Green Channel 2']
                st.image(data_g2, caption="Green Channel 2 (Even Row, Odd Col)", clamp=True, channels='GRAY', use_column_width=True)
                st.pyplot(plot_histogram(data_g2, "Green Channel 2", color='darkgreen'))

else:
    st.info("👈 请在左侧上传 .raw 文件以开始分析")