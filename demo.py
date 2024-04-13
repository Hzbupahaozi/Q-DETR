import torch

torch.manual_seed(42)

# 创建一个形状为(3, 2, 3)的张量
tensor1 = torch.randn(3, 2, 3)
tensor2 = torch.randn(3, 2, 3)
print(tensor1)
print(tensor2)
# 将张量的数据结构转换为(2, 3, 3)
reshaped_tensor1 = tensor1.permute(1, 2, 0)
reshaped_tensor2 = tensor2.permute(1, 2, 0)
# print(reshaped_tensor)

norm_tensor_transpose1 = reshaped_tensor1.permute(0, 2, 1)
norm_tensor_transpose2 = reshaped_tensor2.permute(0, 2, 1)
print(norm_tensor_transpose1)
print(norm_tensor_transpose2)
print(norm_tensor_transpose1-norm_tensor_transpose2)
# # 对后两个维度求二范数，得到形状为(2, 1)的张量
# norm_tensor = torch.norm(norm_tensor_transpose, p=2, dim=(1, 2)).unsqueeze(1)
#
# distill_loss = torch.mean(norm_tensor)
# print(norm_tensor.shape)
# print(norm_tensor)
# print(distill_loss)
