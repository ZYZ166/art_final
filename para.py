# ─── 超参数配置 ───────────────────────────────────────────────────────────────────

# 图片尺寸
# 内容图保留原始分辨率（分辨率越高显存/内存占用越大，CPU 慎用超大图）
# 风格图会被缩放到与内容图相同的尺寸，IMAGE_SIZE 不再使用

# 优化迭代次数（LBFGS 以 closure 调用次数计）
NUM_STEPS = 500

# 损失权重
CONTENT_WEIGHT = 1          # 内容保留强度
STYLE_WEIGHT   = 10_000_000  # 风格迁移强度（越大风格越浓）

# LBFGS 学习率（通常固定为 1.0）
LEARNING_RATE = 1.0

# VGG19 特征层选择（conv_N 表示 VGG19 中第 N 个卷积层）
# 内容层：conv_9（第4块第1个卷积，捕捉高层语义结构）
CONTENT_LAYERS = ['conv_9']
# 风格层：每块的第1个卷积，从低层纹理到高层风格均有覆盖
STYLE_LAYERS   = ['conv_1', 'conv_3', 'conv_5', 'conv_9', 'conv_13']

# 路径
CONTENT_PATH = 'inputs/content.jpg'
STYLE_PATH   = 'inputs/style.jpg'
OUTPUT_PATH  = 'outputs/result.jpg'

# ─── 渐变预览 / GIF ──────────────────────────────────────────────────────────────

# 每隔多少步更新一次预览窗口并保存一帧（0 = 关闭预览）
PREVIEW_INTERVAL = 10

# 是否在完成后将中间帧合成为 GIF 动图
MAKE_GIF  = True
GIF_PATH  = 'outputs/progress.gif'
