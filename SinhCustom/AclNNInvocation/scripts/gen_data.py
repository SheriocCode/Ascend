#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright 2022-2023 Huawei Technologies Co., Ltd
import numpy as np
import os
def gen_golden_data_simple():
    input_dir = "./input"
    output_dir = "./output"
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    input_x = np.random.uniform(1, 10, [8, 2048]).astype(np.float16)
    golden = np.sinh(input_x)

    input_x.tofile(os.path.join(input_dir, "input_x.bin"))
    golden.tofile(os.path.join(output_dir, "golden.bin"))

if __name__ == "__main__":
    gen_golden_data_simple()
