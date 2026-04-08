# nst.py - 神经网络风格迁移 (Neural Style Transfer)
# 基于 Gatys et al. 2015《A Neural Algorithm of Artistic Style》
# 使用预训练 VGG19 提取特征，LBFGS 优化器迭代合成图像

import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')          # 弹出独立窗口；若报错可改 'Qt5Agg' 或 'Agg'
import matplotlib.pyplot as plt

import para

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'运行设备: {device}{"（GPU加速）" if device.type == "cuda" else "（CPU，如较慢可减小 IMAGE_SIZE）"}')


# ─── 图片加载与保存 ──────────────────────────────────────────────────────────────

def load_image(path, size=None):
    """加载图片并归一化为 VGG19 所需格式。
    size=None 保留原始分辨率；size=(h,w) 或 size=n（正方形）则缩放。"""
    image = Image.open(path).convert('RGB')
    ops = []
    if size is not None:
        ops.append(transforms.Resize(size if isinstance(size, tuple) else (size, size)))
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]
    return transforms.Compose(ops)(image).unsqueeze(0).to(device)


def save_image(tensor, path):
    """反归一化张量并保存为图片文件"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
    img = tensor.squeeze(0).clone().detach()
    img = (img * std + mean).clamp(0, 1)
    transforms.ToPILImage()(img.cpu()).save(path)


def tensor_to_pil(tensor):
    """张量转 PIL Image（用于预览和 GIF）"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
    img  = tensor.squeeze(0).clone().detach()
    img  = (img * std + mean).clamp(0, 1)
    return transforms.ToPILImage()(img.cpu())


def make_gif(frames, path, duration=120):
    """将 PIL Image 列表合成 GIF，duration 单位为毫秒/帧"""
    if not frames:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=duration,
    )
    print(f'GIF 已保存至 {path}（{len(frames)} 帧）')


# ─── 损失模块 ────────────────────────────────────────────────────────────────────

class ContentLoss(nn.Module):
    """内容损失：当前特征与内容图特征的 MSE"""
    def __init__(self, target):
        super().__init__()
        self.target = target.detach()
        self.loss = torch.tensor(0.0, device=device)

    def forward(self, x):
        self.loss = nn.functional.mse_loss(x, self.target)
        return x


class StyleLoss(nn.Module):
    """风格损失：当前 Gram 矩阵与风格图 Gram 矩阵的 MSE
    Gram 矩阵捕捉特征通道间的相关性，对应纹理/风格信息"""
    def __init__(self, target_feature):
        super().__init__()
        self.target = self._gram(target_feature).detach()
        self.loss = torch.tensor(0.0, device=device)

    @staticmethod
    def _gram(x):
        b, c, h, w = x.size()
        f = x.view(c, h * w)
        return torch.mm(f, f.t()) / (c * h * w)  # 归一化避免大尺寸梯度爆炸

    def forward(self, x):
        self.loss = nn.functional.mse_loss(self._gram(x), self.target)
        return x


# ─── 构建插入损失层的模型 ─────────────────────────────────────────────────────────

def build_model(content_img, style_img):
    """遍历 VGG19，在指定卷积层后插入损失模块，返回截断模型和损失列表"""
    vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()
    for p in vgg.parameters():
        p.requires_grad_(False)

    model = nn.Sequential()
    content_losses = []
    style_losses   = []
    conv_idx = 0

    for layer in vgg.children():
        if isinstance(layer, nn.Conv2d):
            conv_idx += 1
            name = f'conv_{conv_idx}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{conv_idx}'
            layer = nn.ReLU(inplace=False)  # 原地操作会破坏反向传播
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{conv_idx}'
        else:
            name = f'bn_{conv_idx}'

        model.add_module(name, layer)

        # 在指定内容层后插入 ContentLoss
        if name in para.CONTENT_LAYERS:
            cl = ContentLoss(model(content_img))
            model.add_module(f'content_loss_{conv_idx}', cl)
            content_losses.append(cl)

        # 在指定风格层后插入 StyleLoss
        if name in para.STYLE_LAYERS:
            sl = StyleLoss(model(style_img))
            model.add_module(f'style_loss_{conv_idx}', sl)
            style_losses.append(sl)

    # 截断到最后一个损失层，避免通过后续层做无用前向传播
    modules = list(model.children())
    last_loss_idx = max(
        i for i, m in enumerate(modules)
        if isinstance(m, (ContentLoss, StyleLoss))
    )
    model = model[:last_loss_idx + 1]

    return model, content_losses, style_losses


# ─── 风格迁移主循环 ───────────────────────────────────────────────────────────────

def run_nst(content_img, style_img):
    print('构建模型，计算目标特征...')
    model, content_losses, style_losses = build_model(content_img, style_img)

    # 从内容图开始优化（比随机噪声收敛更快）
    input_img = content_img.clone().requires_grad_(True)
    optimizer = optim.LBFGS([input_img], lr=para.LEARNING_RATE)

    print(f'开始迭代，共 {para.NUM_STEPS} 步...\n'
          f'  {"步数":>6}  {"内容损失":>12}  {"风格损失":>12}  {"耗时":>6}')
    print('  ' + '-' * 46)

    # ── 预览窗口初始化 ──
    preview_on = para.PREVIEW_INTERVAL > 0
    frames = []   # 收集中间帧，用于生成 GIF

    if preview_on:
        plt.ion()
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.suptitle('Neural Style Transfer — 实时进度')
        for ax, title in zip(axes, ['内容图', '风格图', '当前结果']):
            ax.set_title(title)
            ax.axis('off')
        axes[0].imshow(tensor_to_pil(content_img))
        axes[1].imshow(tensor_to_pil(style_img))
        im_result = axes[2].imshow(tensor_to_pil(input_img))
        plt.tight_layout()
        plt.pause(0.01)

    start = time.time()
    step  = [0]

    while step[0] < para.NUM_STEPS:
        def closure():
            with torch.no_grad():
                input_img.clamp_(-3.0, 3.0)
            optimizer.zero_grad()
            model(input_img)

            content_score = sum(cl.loss for cl in content_losses) * para.CONTENT_WEIGHT
            style_score   = sum(sl.loss for sl in style_losses)   * para.STYLE_WEIGHT
            loss = content_score + style_score
            loss.backward()

            step[0] += 1

            # 每隔 PREVIEW_INTERVAL 步：刷新窗口 + 收集帧
            if preview_on and step[0] % para.PREVIEW_INTERVAL == 0:
                pil = tensor_to_pil(input_img)
                frames.append(pil)
                im_result.set_data(pil)
                fig.suptitle(f'Neural Style Transfer — 第 {step[0]}/{para.NUM_STEPS} 步')
                plt.pause(0.001)

            if step[0] % 50 == 0:
                elapsed = time.time() - start
                print(f'  {step[0]:>5}/{para.NUM_STEPS}'
                      f'  {content_score.item():>12.2f}'
                      f'  {style_score.item():>12.2f}'
                      f'  {elapsed:>5.0f}s')
            return loss

        optimizer.step(closure)

    if preview_on:
        plt.ioff()
        plt.show(block=False)

    elapsed = time.time() - start
    print(f'\n完成！总耗时 {elapsed:.1f}s')
    return input_img, frames


# ─── 入口 ────────────────────────────────────────────────────────────────────────

def main():
    content_img = load_image(para.CONTENT_PATH)          # 原始分辨率
    _, _, h, w  = content_img.shape
    print(f'内容图原始尺寸: {w}x{h}px')

    style_img = load_image(para.STYLE_PATH, size=(h, w)) # 风格图缩放到一致

    result, frames = run_nst(content_img, style_img)

    save_image(result, para.OUTPUT_PATH)
    print(f'结果已保存至 {para.OUTPUT_PATH}')

    if para.MAKE_GIF and frames:
        frames.append(tensor_to_pil(result))   # 末尾加一帧最终结果
        make_gif(frames, para.GIF_PATH)

    if para.PREVIEW_INTERVAL > 0:
        input('按 Enter 关闭预览窗口...')
        plt.close('all')


if __name__ == '__main__':
    main()
