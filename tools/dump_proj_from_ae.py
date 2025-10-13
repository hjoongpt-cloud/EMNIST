# Extract proj_down weights from AE checkpoint and slice by keep_idx
# Saves to .npz with keys: W (64, KEPT, 1, 1), b (64,)

import argparse, numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument('--ae_ckpt', required=True, help='Path to ae9x9_proj64.pt')
ap.add_argument('--keep_idx', required=True, help='Path to keep indices .npy (e.g., prune_step*_keep_indices.npy)')
ap.add_argument('--out', required=True, help='Output .npz path (e.g., proj_init_k96.npz)')

args = ap.parse_args()

ck = torch.load(args.ae_ckpt, map_location='cpu')
# handle both plain state_dict or nested under 'state_dict'
sd = ck if isinstance(ck, dict) and any(k.startswith('proj_down.') for k in ck.keys()) else ck.get('state_dict', {})
if not any(k.startswith('proj_down.') for k in sd.keys()):
    raise RuntimeError('proj_down.* not found in checkpoint state_dict')

W = sd['proj_down.weight'].cpu().numpy()  # (64,K,1,1)
B = sd['proj_down.bias'].cpu().numpy()    # (64,)
keep = np.load(args.keep_idx)

W = W[:, keep, :, :]
np.savez(args.out, W=W.astype('float32'), b=B.astype('float32'))
print('saved', args.out, W.shape, B.shape)
