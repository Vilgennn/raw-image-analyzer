import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 核心逻辑函数
# ==========================================

def detect_shape_from_size(file_bytes_len):
    """基于文件大小的自动分辨率推断"""
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
    """读取 RAW 数据"""
    bytes_data = uploaded_file.getvalue()
    file_size = len(bytes_data)
    
    final_w, final_h, final_depth = None, None, 8
    
    if width_input > 0 and height_input > 0:
        final_w, final_h = width_input, height_input
        final_depth = 16 if file_size >= final_w * final_h * 2 else 8
        st.info(f"使用手动输入分辨率: {final_w} x {final_h} ({final_depth}-bit)")
    else:
        w_auto, h_auto, depth_auto = detect_shape_from_size(file_size)
        if w_auto:
            final_w, final_h, final_depth = w_auto, h_auto, depth_auto
            st.success(f"自动推断分辨率成功: {final_w} x {final_h} ({final_depth}-bit)")
        else:
            st.error(f"无法自动推断分辨率 (文件大小: {file_size})。请手动输入。")
            return None

    dtype = np.uint16 if final_depth > 8 else np.uint8
    try:
        data = np.frombuffer(bytes_data, dtype=dtype)
        expected_pixels = final_w * final_h
        if len(data) < expected_pixels:
            st.error(f"数据不足。需要 {expected_pixels}，实际 {len(data)}。")
            return None
        data = data[:expected_pixels].reshape((final_h, final_w))
        return data
    except Exception as e:
        st.error(f"解析出错: {e}")
        return None

def split_channels_strict(bayer_data):
    """严格按照用户指定映射分离通道"""
    # 使用 astype(float) 防止计算统计量时溢出
    g1 = bayer_data[0::2, 0::2]
    r = bayer_data[0::2, 1::2]
    b = bayer_data[1::2, 0::2]
    g2 = bayer_data[1::2, 1::2]
    
    return {
        'Green Channel 1': g1,
        'Red Channel': r,
        'Blue Channel': b,
        'Green Channel 2': g2
    }

def plot_histogram(data, title, color='gray'):
    """绘制 Matplotlib 直方图"""
    fig, ax = plt.subplots(figsize=(5, 3))
    # 展平数据并绘制
    ax.hist(data.ravel(), bins=50, color=color, alpha=0.7, density=False)
    ax.set_title(f"{title} Histogram")
    ax.set_xlabel("Pixel Value (DN)")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    return fig

# ==========================================
# Streamlit 页面布局
# ==========================================

st.set_page_config(page_title="Raw Bayer Analyzer", layout="wide")

st.title("🔬 Naked Binary RAW 图像分析工具")
st.markdown("""
支持 **BRGB** 排列模式 (Odd: B R, Even: G B)。
**新增功能**：统计表中包含积分密度（总光强），用于能量分析。
""")

with st.sidebar:
    st.header("1. 配置与上传")
    input_width = st.number_input("Width (px)", min_value=0, value=0, step=1)
    input_height = st.number_input("Height (px)", min_value=0, value=0, step=1)
    st.divider()
    uploaded_file = st.file_uploader("上传 .raw 文件", type=['raw', 'bin'])

if uploaded_file is not None:
    raw_img = load_raw_data(uploaded_file, input_width, input_height)
    
    if raw_img is not None:
        channels = split_channels_strict(raw_img)
        
        # === 3. 计算统计数据 (含积分) ===
        stats_list = []
        
        # 辅助函数：生成单行统计
        def get_stats(name, arr):
            # 转换为 float64 以计算总和，防止 uint8/16 溢出
            arr_float = arr.astype(np.float64)
            return {
                "Channel": name,
                "Mean": float(np.mean(arr_float)),
                "Std Dev": float(np.std(arr_float)),
                "Min": int(np.min(arr)),
                "Max": int(np.max(arr)),
                # 积分信息：所有像素值的总和 (Total Energy/Flux)
                "Integrated Density (Sum)": float(np.sum(arr_float)) 
            }
        
        # 原始图像
        stats_list.append(get_stats("Original RAW", raw_img))
        # 各通道
        for name, data in channels.items():
            stats_list.append(get_stats(name, data))
            
        df_stats = pd.DataFrame(stats_list)
        
        # === 显示区域 ===
        
        st.header("2. 统计信息概览")
        st.caption("Integrated Density (Sum) 代表该通道所有像素灰度值的总和，反映了总接收光能量。")
        
        # 使用 column_config 优化大数显示
        st.dataframe(
            df_stats,
            use_container_width=True,
            column_config={
                "Channel": "通道名称",
                "Mean": st.column_config.NumberColumn("均值", format="%.2f"),
                "Std Dev": st.column_config.NumberColumn("标准差", format="%.2f"),
                "Min": st.column_config.NumberColumn("最小值"),
                "Max": st.column_config.NumberColumn("最大值"),
                "Integrated Density (Sum)": st.column_config.NumberColumn(
                    "积分密度 (Sum)", 
                    help="直方图积分信息：所有像素灰度值之和",
                    format="%d"  # 使用科学计数法显示，例如 1.23e+07
                )
            }
        )
        
        st.divider()
        st.header("3. 可视化与直方图")
        
        tab_raw, tab_channels = st.tabs(["原始 RAW 图像", "分离通道 (R/B/G1/G2)"])
        
        with tab_raw:
            col1, col2 = st.columns([1, 1])
            with col1:
                st.subheader("原始灰度图")
                st.image(raw_img, caption=f"Original RAW ({raw_img.shape[1]}x{raw_img.shape[0]})", clamp=True, channels='GRAY', use_column_width=True)
            with col2:
                st.subheader("灰度直方图")
                st.pyplot(plot_histogram(raw_img, "Original RAW", color='black'))

        with tab_channels:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("### 🔴 Red Channel")
                data_r = channels['Red Channel']
                st.image(data_r, caption="Red Channel", clamp=True, channels='GRAY', use_column_width=True)
                st.pyplot(plot_histogram(data_r, "Red Channel", color='red'))
            
            with c2:
                st.markdown("### 🔵 Blue Channel")
                data_b = channels['Blue Channel']
                st.image(data_b, caption="Blue Channel", clamp=True, channels='GRAY', use_column_width=True)
                st.pyplot(plot_histogram(data_b, "Blue Channel", color='blue'))
                
            st.divider()
            c3, c4 = st.columns(2)
            with c3:
                st.markdown("### 🟢 Green Channel 1")
                data_g1 = channels['Green Channel 1']
                st.image(data_g1, caption="Green Channel 1", clamp=True, channels='GRAY', use_column_width=True)
                st.pyplot(plot_histogram(data_g1, "Green Channel 1", color='green'))

            with c4:
                st.markdown("### 🟢 Green Channel 2")
                data_g2 = channels['Green Channel 2']
                st.image(data_g2, caption="Green Channel 2", clamp=True, channels='GRAY', use_column_width=True)
                st.pyplot(plot_histogram(data_g2, "Green Channel 2", color='darkgreen'))

else:
    st.info("👈 请在左侧上传 .raw 文件")
