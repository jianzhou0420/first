# import torch
# import torch.nn as nn

# # 设置一个固定的随机种子，以确保每次运行结果都一样
# torch.manual_seed(42)

# model = nn.Sequential(
#     nn.Linear(10, 8),  # Layer A (model[0])
#     nn.Linear(8, 5),   # Layer B (model[1])
#     nn.Linear(5, 2)    # Layer C (model[2])
# )

# # 冻结中间层 Layer B
# print("--- 冻结中间层 Layer B ---")
# for param in model[1].parameters():
#     param.requires_grad = False

# # 1. 初始化优化器 (关键步骤)
# # 我们只将 requires_grad=True 的参数传递给优化器
# trainable_params = filter(lambda p: p.requires_grad, model.parameters())
# optimizer = torch.optim.SGD(trainable_params, lr=0.1)  # 使用稍大的学习率以看清变化

# print("\n--- 初始权重 (训练前) ---")
# print(f"Layer A 的初始权重:\n{model[0].weight.data}\n")
# print(f"Layer B 的初始权重 (被冻结，将保持不变):\n{model[1].weight.data}\n")
# print(f"Layer C 的初始权重:\n{model[2].weight.data}\n")


# # 2. 模拟完整的训练步骤
# input_tensor = torch.randn(1, 10)
# target = torch.tensor([[1.0, 0.0]])

# # 2a. 清除旧的梯度
# optimizer.zero_grad()

# # 2b. 前向传播和计算 Loss
# output = model(input_tensor)
# loss = nn.MSELoss()(output, target)

# # 2c. 反向传播计算梯度
# loss.backward()

# # 2d. 调用优化器更新权重 (!!!)
# optimizer.step()


# # 3. 检查更新后的权重
# print("\n--- 调用 optimizer.step() 后的权重 ---")

# # Layer A
# print(f"\nLayer A 的更新后权重 (已改变):\n{model[0].weight.data}")

# # Layer B
# print(f"\nLayer B 的更新后权重 (无变化):\n{model[1].weight.data}")

# # Layer C
# print(f"\nLayer C 的更新后权重 (已改变):\n{model[2].weight.data}")
import torch
import torch.nn as nn
import numpy as np

# 为了保证每次运行结果一致，设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# --- 第 1 部分：使用 PyTorch 自动计算梯度 ---

print("=" * 50)
print("Part 1: Calculating Gradients with PyTorch")
print("=" * 50)

# 1. 定义网络结构
# 一个简单的三层线性网络：输入层 -> 隐藏层1 -> 隐藏层2 -> 输出层
# 使用 ReLU 作为激活函数


class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden2_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out


# 2. 初始化网络、损失函数和数据
input_size = 10
hidden1_size = 5
hidden2_size = 3
output_size = 1
batch_size = 1  # 使用单个样本以便于手动跟踪

model = SimpleNet(input_size, hidden1_size, hidden2_size, output_size)
criterion = nn.MSELoss()  # 使用均方误差作为损失函数

# 创建一个随机输入样本和目标值
x = torch.randn(batch_size, input_size)
y_true = torch.randn(batch_size, output_size)

# 3. 执行前向传播和反向传播
# 清除旧的梯度
model.zero_grad()

# 前向传播
y_pred = model(x)

# 计算损失
loss = criterion(y_pred, y_true)

# 反向传播，PyTorch 会自动计算所有可训练参数的梯度
loss.backward()

# 4. 打印 PyTorch 计算的梯度
print("\n--- PyTorch Gradients ---")
print("Gradient for fc3.weight:\n", model.fc3.weight.grad)
print("\nGradient for fc3.bias:\n", model.fc3.bias.grad)
print("\nGradient for fc2.weight:\n", model.fc2.weight.grad)
print("\nGradient for fc2.bias:\n", model.fc2.bias.grad)
print("\nGradient for fc1.weight:\n", model.fc1.weight.grad)
print("\nGradient for fc1.bias:\n", model.fc1.bias.grad)


# --- 第 2 部分：使用 NumPy 手动复现梯度计算 ---

print("\n" + "=" * 50)
print("Part 2: Manually Replicating Gradients with NumPy")
print("=" * 50)

# 1. 从 PyTorch 模型中提取权重、偏置和中间变量，并转换为 NumPy 数组
# 这是为了确保我们的手动计算使用与 PyTorch 完全相同的初始值
W1 = model.fc1.weight.data.numpy()
b1 = model.fc1.bias.data.numpy().reshape(1, -1)
W2 = model.fc2.weight.data.numpy()
b2 = model.fc2.bias.data.numpy().reshape(1, -1)
W3 = model.fc3.weight.data.numpy()
b3 = model.fc3.bias.data.numpy().reshape(1, -1)

x_np = x.data.numpy()
y_true_np = y_true.data.numpy()

# 2. 手动执行前向传播，并保存所有中间结果，因为反向传播需要它们
# Layer 1
z1 = x_np @ W1.T + b1
a1 = np.maximum(0, z1)  # ReLU activation

# Layer 2
z2 = a1 @ W2.T + b2
a2 = np.maximum(0, z2)  # ReLU activation

# Layer 3 (Output)
z3 = a2 @ W3.T + b3
y_pred_np = z3

# 3. 手动执行反向传播（链式法则）

# 起点：计算损失函数关于最终输出的梯度
# loss = (y_pred - y_true)^2, 导数是 2 * (y_pred - y_true)
# 因为 PyTorch 的 MSELoss 是 (1/N) * sum(...), N=1, 所以梯度就是 (y_pred - y_true)
# update: pytorch's MSELoss is mean, which is sum/n, n is the number of elements in the tensor.
# for a tensor of size (B, C), n = B*C
# In our case, y_pred is (1,1), so n=1.
# But if output_size was > 1, the gradient would be (2/n) * (y_pred - y_true)
grad_y_pred = 2 * (y_pred_np - y_true_np) / y_pred_np.size


# --- Backprop through Layer 3 (fc3) ---
# z3 = a2 @ W3.T + b3
# d(loss)/d(W3) = d(loss)/d(z3) * d(z3)/d(W3)
# d(loss)/d(z3) is grad_y_pred because y_pred = z3
# d(z3)/d(W3) is a2
grad_z3 = grad_y_pred
grad_W3 = grad_z3.T @ a2
grad_b3 = np.sum(grad_z3, axis=0)

# 传递到上一层的梯度
# d(loss)/d(a2) = d(loss)/d(z3) * d(z3)/d(a2)
# d(z3)/d(a2) is W3
grad_a2 = grad_z3 @ W3

# --- Backprop through Layer 2 (relu2 and fc2) ---
# a2 = relu(z2)
# The derivative of ReLU is 1 if input > 0, else 0.
grad_z2 = grad_a2 * (z2 > 0)

# z2 = a1 @ W2.T + b2
# d(loss)/d(W2) = d(loss)/d(z2) * d(z2)/d(W2)
grad_W2 = grad_z2.T @ a1
grad_b2 = np.sum(grad_z2, axis=0)

# d(loss)/d(a1) = d(loss)/d(z2) * d(z2)/d(a1)
grad_a1 = grad_z2 @ W2

# --- Backprop through Layer 1 (relu1 and fc1) ---
# a1 = relu(z1)
grad_z1 = grad_a1 * (z1 > 0)

# z1 = x_np @ W1.T + b1
# d(loss)/d(W1) = d(loss)/d(z1) * d(z1)/d(W1)
grad_W1 = grad_z1.T @ x_np
grad_b1 = np.sum(grad_z1, axis=0)


# 4. 打印手动计算的 NumPy 梯度
print("\n--- NumPy Manual Gradients ---")
print("Gradient for W3:\n", grad_W3)
print("\nGradient for b3:\n", grad_b3)
print("\nGradient for W2:\n", grad_W2)
print("\nGradient for b2:\n", grad_b2)
print("\nGradient for W1:\n", grad_W1)
print("\nGradient for b1:\n", grad_b1)


# --- 第 3 部分：对比结果 ---
print("\n" + "=" * 50)
print("Part 3: Comparing Gradients")
print("=" * 50)

# 使用 np.allclose 来比较浮点数数组是否在容差范围内相等
print("fc3.weight gradients match:", np.allclose(model.fc3.weight.grad.numpy(), grad_W3))
print("fc3.bias gradients match:  ", np.allclose(model.fc3.bias.grad.numpy(), grad_b3))
print("fc2.weight gradients match:", np.allclose(model.fc2.weight.grad.numpy(), grad_W2))
print("fc2.bias gradients match:  ", np.allclose(model.fc2.bias.grad.numpy(), grad_b2))
print("fc1.weight gradients match:", np.allclose(model.fc1.weight.grad.numpy(), grad_W1))
print("fc1.bias gradients match:  ", np.allclose(model.fc1.bias.grad.numpy(), grad_b1))
