import torch

input_path = 'weights/scannet200-backbone-only.pth'
output_path = 'weights/scannet200-backbone-3ch.pth'

print(f"Učitavam: {input_path}")
ckpt = torch.load(input_path, map_location='cpu')

if 'state_dict' in ckpt:
    state_dict = ckpt['state_dict']
elif 'model' in ckpt:
    state_dict = ckpt['model']
else:
    state_dict = ckpt

# 1. Dinamičko traženje pravog ključa
target_key = None
for k in state_dict.keys():
    if 'stem.conv.weight' in k:
        target_key = k
        break

if target_key is None:
    print("GREŠKA: Ne mogu pronaći sloj!")
    # Ispisujemo prvih par ključeva da vidimo strukturu ako opet pukne
    print("Dostupni ključevi u datoteci:", list(state_dict.keys())[:5])
else:
    print(f"Pronađen točan ključ: {target_key}")
    stari_weight = state_dict[target_key]
    
    # 2. Režemo zadnju dimenziju (s kanala 6 na kanal 3)
    novi_weight = stari_weight[..., :3] 
    
    # 3. Mijenjamo stari sloj novim
    state_dict[target_key] = novi_weight
    
    print(f"Stari oblik sloja: {stari_weight.shape}")
    print(f"Novi oblik sloja:  {novi_weight.shape}")
    
    torch.save(ckpt, output_path)
    print(f"Gotovo! Nova arhitektura spremljena u: {output_path}")