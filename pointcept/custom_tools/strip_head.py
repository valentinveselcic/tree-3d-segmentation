import torch

input_path = 'weights/model_best_scannet200.pth'
output_path = 'weights/scannet200-backbone-only.pth'

print(f"Učitavam: {input_path}")
ckpt = torch.load(input_path, map_location='cpu')

# Pointcept obično sprema težine unutar ključa 'state_dict' ili 'model'
if 'state_dict' in ckpt:
    state_dict = ckpt['state_dict']
elif 'model' in ckpt:
    state_dict = ckpt['model']
else:
    state_dict = ckpt

# Pronađi i obriši sve slojeve koji u imenu imaju 'seg_head'
keys_to_delete = [k for k in state_dict.keys() if 'seg_head' in k]

for k in keys_to_delete:
    print(f"Brišem sloj: {k}")
    del state_dict[k]

# Spremi natrag na disk
torch.save(ckpt, output_path)
print(f"Gotovo! Čiste backbone težine spremljene u: {output_path}")