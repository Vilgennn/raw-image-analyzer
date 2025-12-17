import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rawpy  # 需要安装: pip install rawpy

# ==========================================
# 核心逻辑函数
# ==========================================

def detect_shape_from_size(file_bytes_len):
    """基于文件大小的自动分辨率推断 (仅用于 Naked Binary)"""
    common_resolutions = [
        (2592, 1944), # 5MP
        (1920, 1080), # 1080p
        (1280, 720),  # 720p
        (3264, 2448), # 8MP
        (4000, 3000), # 12MP
        (4608, 3456), # 16MP
        (6000, 4000), # 24MP (常见)
        (7952, 5304), # 42MP (Sony A7R等)
        (9504, 6336), # 60MP (Sony A7R4等)
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
    """读取 RAW 数据 (支持 .raw/.bin 二进制流 和 .ARW 格式)"""
    filename = uploaded_file.name.lower()
    
    # === 分支 1: 处理 Sony .ARW 格式 ===
    if filename.endswith('.arw'):
        try:
            # rawpy 可以直接处理文件对象
            with rawpy.imread(uploaded_file) as raw:
                # 获取原始 Bayer 数据 (2D array)
                data = raw.raw_image.copy()
                h, w = data.shape
                
                # 获取 Bayer 模式信息 (用于提示)
                pattern_name = raw.color_desc.decode('utf-8') # 通常是 RGBG
                st.success(f"成功解析 ARW 文件: {w} x {h} | Bayer Pattern: {pattern_name}")
                st.info("注意：ARW 文件的 Bayer 排列通常为 RGGB。请检查下方通道分离是否符合预期。")
                return data
        except Exception as e:
            st.error(f"ARW 解析失败: {e}")
            return None

    # === 分支 2: 处理 Naked Binary (.raw / .bin) ===
    else:
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
                st.success(f"自动推断二进制分辨率成功: {final_w} x {final_h} ({final_depth}-bit)")
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
            st.error(f"二进制解析出错: {e}")
            return None

def split_channels_strict(bayer_data):
    """严格按照用户指定映射分离通道"""
    # 注意：这假设了特定的 Bayer 排列 (例如 GRBG 或 RGGB 的某种变体)
    # 对于 ARW (通常是 RGGB)，这里的映射可能需要用户脑补转换
    # 0,0 -> G1 / 0,1 -> R / 1,0 -> B / 1,1 -> G2 (这是原代码逻辑)
    
    # 使用 astype(float) 防止计算统计量时溢出
    g1 = bayer_data[0::2, 0::2]
    r = bayer_data[0::2, 1::2]
    b = bayer_data[1::2, 0::2]
    g2 = bayer_data[1::2, 1::2]
    
    return {
        'Channel (0,0) [Ex: G1]': g1,
        'Channel (0,1) [Ex: R]': r,
        'Channel (1,0) [Ex: B]': b,
        'Channel (1,1) [Ex: G2]': g2
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

st.title("🔬 Universal RAW 图像分析工具")
st.markdown("""
支持 **Naked Binary (.raw/.bin)** 及 **Sony ARW** 格式。
**统计功能**：包含积分密度（总光强），用于能量分析。
""")

with st.sidebar:
    st.header("1. 配置与上传")
    st.caption("对于 .ARW 文件，无需手动输入分辨率。")
    input_width = st.number_input("Width (px) [仅用于 .raw]", min_value=0, value=0, step=1)
    input_height = st.number_input("Height (px) [仅用于 .raw]", min_value=0, value=0, step=1)
    st.divider()
    # 增加 arw 支持
    uploaded_file = st.file_uploader("上传文件", type=['raw', 'bin', 'arw'])

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
        st.caption("Integrated Density (Sum) 代表该通道所有像素灰度值的总和。")
        
        # 使用 column_config 优化大数显示
        st.dataframe(
            df_stats,
            use_container_width=True,
            column_config={
                "Channel": "通道位置 / 预期名称",
                "Mean": st.column_config.NumberColumn("均值", format="%.2f"),
                "Std Dev": st.column_config.NumberColumn("标准差", format="%.2f"),
                "Min": st.column_config.NumberColumn("最小值"),
                "Max": st.column_config.NumberColumn("最大值"),
                "Integrated Density (Sum)": st.column_config.NumberColumn(
                    "积分密度 (Sum)", 
                    help="直方图积分信息：所有像素灰度值之和",
                    format="%d"  # 使用科学计数法或整型显示
                )
            }
        )
        
        st.divider()
        st.header("3. 可视化与直方图")
        
        tab_raw, tab_channels = st.tabs(["原始 RAW 图像", "分离通道 (Grid View)"])
        
        with tab_raw:
            col1, col2 = st.columns([1, 1])
            with col1:
                st.subheader("原始灰度图")
                # 针对 16bit 数据显示优化，防止全黑
                display_img = raw_img
                if raw_img.dtype == np.uint16:
                    # 简单归一化用于显示，不改变原始数据
                    display_img = (raw_img / 256).astype(np.uint8)
                    st.caption("注：预览图已压缩至 8-bit 显示，统计数据仍基于原始 16-bit")
                
                st.image(display_img, caption=f"RAW Data ({raw_img.shape[1]}x{raw_img.shape[0]})", clamp=True, channels='GRAY', use_column_width=True)
            with col2:
                st.subheader("灰度直方图")
                st.pyplot(plot_histogram(raw_img, "Original RAW", color='black'))

        with tab_channels:
            st.info("注意：如果上传的是 ARW (RGGB)，通常 (0,0)=R, (0,1)=G, (1,0)=G, (1,1)=B。请根据实际情况对应下方图表。")
            
            c1, c2 = st.columns(2)
            with c1:
                # 0,0
                name = 'Channel (0,0) [Ex: G1]'
                st.markdown(f"### ↖️ {name}")
                data_sub = channels[name]
                st.image(data_sub, caption=name, clamp=True, channels='GRAY', use_column_width=True)
                st.pyplot(plot_histogram(data_sub, name, color='green'))
            
            with c2:
                # 0,1
                name = 'Channel (0,1) [Ex: R]'
                st.markdown(f"### ↗️ {name}")
                data_sub = channels[name]
                st.image(data_sub, caption=name, clamp=True, channels='GRAY', use_column_width=True)
                st.pyplot(plot_histogram(data_sub, name, color='red'))
                
            st.divider()
            c3, c4 = st.columns(2)
            with c3:
                # 1,0
                name = 'Channel (1,0) [Ex: B]'
                st.markdown(f"### ↙️ {name}")
                data_sub = channels[name]
                st.image(data_sub, caption=name, clamp=True, channels='GRAY', use_column_width=True)
                st.pyplot(plot_histogram(data_sub, name, color='blue'))

            with c4:
                # 1,1
                name = 'Channel (1,1) [Ex: G2]'
                st.markdown(f"### ↘️ {name}")
                data_sub = channels[name]
                st.image(data_sub, caption=name, clamp=True, channels='GRAY', use_column_width=True)
                st.pyplot(plot_histogram(data_sub, name, color='darkgreen'))

else:
    st.info("👈 请在左侧上传 .raw, .bin 或 .arw 文件")
