# -*- coding: utf-8 -*-
# Ciallo～(∠・ω )⌒☆

import torch
print("CUDA可用:", torch.cuda.is_available())
print("CUDA版本:", torch.version.cuda)
print("GPU设备数:", torch.cuda.device_count())
if torch.cuda.is_available():
	print("当前GPU:", torch.cuda.current_device())
	print("GPU名称:", torch.cuda.get_device_name())