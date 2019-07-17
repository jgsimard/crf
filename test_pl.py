import numpy as np
from PIL import Image
import torch

from permuthohedral_lattice import PermutohedralLattice


img = np.asarray(Image.open("small_input.bmp"))
# img = np.asarray(Image.open("gray_original.bmp"))

print(img.shape)

indices = np.reshape(np.indices(img.shape[:2]), (2, -1))[None, :]
img = np.transpose(img, (2, 0, 1))
rgb = np.reshape(img, (3, -1))[None, :]
# gray= np.reshape(img, (1, -1))[None, :]

print(indices.shape)
# rgb = np.reshape(np.transpose(img, (2, 0, 1)), (3, -1))[None, :]


pl = PermutohedralLattice.apply

out = pl(torch.from_numpy(indices/1.0).cuda().float(),
         torch.from_numpy(rgb/2).cuda().float())
# out = pl(torch.from_numpy(indices/2.0).cuda().float(),
#          torch.from_numpy(gray/0.125).cuda().float())

output = out.squeeze().cpu().numpy()
output = np.transpose(output, (1, 0))
output = np.reshape(output, (img.shape[1], img.shape[2], 3))
result = Image.fromarray((output/output.max() *255).astype(np.uint8))
result.save('out.png')

print(out.shape, out.max(), out.min())

# Image.fromarray(np.reshape(rgb, img.shape).astype(np.uint8)).save("test_reshape.bmp")
# print(indices.shape, rgb.shape)
