def match_tensor_size(tensor, target_tensor):
    _, _, h, w = tensor.shape
    _, _, th, tw = target_tensor.shape
    dh = h - th
    dw = w - tw
    print(dh, dw)
    print(th, tw)
    print(h, w)
    return dh, dw #tensor[:, :, :h - dh, :w - dw]
