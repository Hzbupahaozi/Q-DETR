# import torch
#
# # 假设 box 是一个张量，数据结构为 {3, 2, 4}
# box = torch.randn(2, 3, 4)
# print(box)
# # 将前两个维度拼接在一起
# new_shape = box.view(2 * 3, 4)
# print(new_shape)
# print(new_shape.shape)


# # 计算GIOU
# import torch
# from torchvision.ops import box_iou
#
# # 假设 box 和 outputs 是给定的张量
# box = torch.randn(300, 4)
# outputs = torch.randn(3, 4)
#
# # 计算 GIOU
# from util import box_ops
#
# loss = torch.diag(box_ops.generalized_box_iou(
#             box_ops.box_cxcywh_to_xyxy(box),
#             box_ops.box_cxcywh_to_xyxy(outputs)))
#
# print(loss)

# # 得到最大的GIOU，比较阈值
# import torch
#
# # 假设 data 是给定的张量，数据结构为 (5, 3)
# data = torch.randn(5, 3)
# print(data)
# for i in range(data.size(1)):  # 遍历每一列
#     max_val = torch.max(data[:, i])  # 找到当前列的最大值
#     threshold = max_val / 2  # 计算该列最大值的一半作为阈值
#
#     for j in range(data.size(0)):  # 遍历列中的每个元素
#         if data[j, i] < threshold:  # 若当前值小于阈值
#             data[j, i] = 0  # 将该元素置零
#
# print(data)
#
# # 判断阈值输出为0
# import torch
#
# # 假设 outputs 和 loss 是给定的张量，数据结构分别为 (300, 4) 和 (300, 3)
# outputs = torch.randn(5, 4)
# loss = torch.randint(0, 2, (5, 3))  # 随机生成 0 或 1 的 loss 数据
# print(outputs)
# print(loss)
# # 判断是否有某行 loss 中包含 0
# mask = torch.any(loss == 0, dim=1)
#
# # 将符合条件的行对应的 outputs 设置为 0
# outputs[mask] = 0
#
# print(outputs)

# import torch
#
# outputs = torch.randn(3, 2, 4)
# # 修改后的outputs
# zero = torch.zeros(3, 4)
# print(zero)
# print(outputs)
# a = object_query1 = (outputs.permute(1, 2, 0)).permute(0, 2, 1)
# object_query1 = (outputs.permute(1, 2, 0)).permute(0, 2, 1)[0]
# object_query2 = (outputs.permute(1, 2, 0)).permute(0, 2, 1)[1]
# print(a)
# print(object_query1)
# print(object_query2)
# stacked_tensor = torch.stack([object_query1, object_query2])
# new_object = stacked_tensor.permute(1, 0, 2)
# print(new_object)
# # mask = torch.any(zero == 0, dim=1)
# # object_query1[mask] = 0
# # print(object_query1)

# 量化特征蒸馏部分代码
# import torch
#
# a = torch.randn(6, 2, 3, 3)
# print(a)
# b = a[5]
# print(b)

# import numpy as np
#
# # 定义数值'a'
# a = 4.2
#
# # 生成数据结构为tensor(256,)，数值为'a'的数据
# data = np.full((256,), a)
#
# print(data)

# # quantnoise
# import torch
#
#
# def round_pass(x):
#     y = x.round()
#     y_grad = x
#     return y.detach() - y_grad.detach() + y_grad  # 得到的数值是y也就是经过round函数，但是求导的是y_grad对x求导为1
#
#
# weight = torch.randn(5, 2, 1)
# alpha = torch.randn(5, 1, 1)
# # weight = torch.randn(1, 10, 2)
# Qn = -8
# Qp = 7
# print(weight)
#
#
# def Quantnoise(input_weight, ratio):
#     mask = torch.rand(input_weight.shape) > ratio
#     print('mask:', mask)
#     no_round_weight = input_weight - mask * input_weight
#     print('不进行量化的权重：', no_round_weight)
#     round_weight = input_weight - no_round_weight
#     print('进行量化的权重：', round_weight)
#
#     w_q = round_pass((round_weight / alpha).clamp(Qn, Qp)) * alpha
#     print('半个权重：', w_q)
#     final_w = w_q + no_round_weight
#     print('整个权重：', final_w)
#
#
# Quantnoise(weight, 0.2)

#
# # 查看pth文件
# import torch  # 命令行是逐行立即执行的
# content = torch.load('./ckpt_2_bit_smca_detr_coco.pth')
# print(content)   # keys()
# # # 之后有其他需求比如要看 key 为 model 的内容有啥
# # print(content['model'])

# # 修改预测框那里有点问题
# import torch
#
# loss_bs1 = torch.randn(10, 3)
# loss_bs2 = torch.randn(10, 4)
# print(loss_bs1)
# size1 = loss_bs1.size(1)
# size2 = loss_bs1.size(0)
# for a in range(size1):
#     max_giou1 = torch.max(loss_bs1[:, a])
#     threshold1 = 0.5 * max_giou1
#     print(threshold1)
#
#     for b in range(size2):
#         if loss_bs1[b, a] < threshold1:
#             loss_bs1[b, a] = 0
# print(loss_bs1)
# # 遍历 Tensor 数据的每一行
# for i in range(loss_bs1.size(0)):
#     row = loss_bs1[i]
#     if torch.all(row == 0):
#         loss_bs2[i] = 0
# print(loss_bs2)

# import torch
#
# box_stu = torch.rand(5, 2)
# box_t = torch.rand(2, 2)
# print(box_stu)
# print(box_t)
# print(box_stu[0])
# # from util import box_ops
# #
# # giou1 = box_ops.generalized_box_iou(
# #     box_ops.box_cxcywh_to_xyxy(box_stu[0]),
# #     box_ops.box_cxcywh_to_xyxy(box_t))
#
# giou1 = torch.rand(1, 2)  # 因为每次只取学生预测框的一个框去和教师的300个预测框对比，所以GIOU应该是1，300（这里用1，3简化一下）
# norm = torch.norm((box_t - box_stu[0]), p=1, dim=1)
# print(giou1)
# print(norm)
# print(giou1 + norm)
# max_norm_index = torch.argmax(giou1 + norm).item()
# print(max_norm_index)


import torch

# 创建原始数据 box
a = torch.randn(1, 4)

# 复制 300 行得到新的数据
expanded_box = a.repeat(300, 1)
print(a)
print(expanded_box[0])

