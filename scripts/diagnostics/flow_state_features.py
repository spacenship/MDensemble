"""Does smoothing the noise starve the flow-state edge features?

The decoder and encoder see the flow state only through three invariant edge
scalars: s_i.u_ij, s_j.u_ij and ||s_j - s_i||. Smoothing makes neighbouring
particles share almost the same s, so the third one should collapse toward
zero -- taking a third of the flow-state signal with it. That would explain
why the correlated-noise arm learns the velocity field measurably worse
(v_cos 0.457 vs 0.695) while scoring better on physics.
"""
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from protein_flow.config import load_config
from protein_flow.data.collate import collate_protein_batch
from protein_flow.data.shard_manifest import ShardManifest
from protein_flow.flow.paths import sample_zero_com_noise
from protein_flow.geometry.graph import build_geometric_graph
from protein_flow.models.geometric_encoder import flow_state_edge_features
from protein_flow.train_rotating import _build_dataset

device = torch.device("cuda:0"); torch.cuda.set_device(device)
torch.cuda.set_per_process_memory_fraction(0.15, device)
config = load_config("configs/mdcath_backbone_rotate_displacement.yaml")
config.data.batch_size = 4; config.data.num_workers = 2
manifest = ShardManifest.load(Path(config.data.rotation.manifest_path))
dd = Path(config.data.mdcath_dir)
files = [p for p in (dd / Path(e.path).name for e in manifest.val_domains) if p.exists()][:2]
ds = _build_dataset(config, dd, files, seed=0, is_validation=True)
loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=2, collate_fn=collate_protein_batch)

raw = next(iter(loader))
batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in raw.items()}
mask, x0 = batch["atom_mask"], batch["source_coords"]
graph = build_geometric_graph(x0, mask, k=config.model.graph.knn_k)
n = x0.shape[0] * x0.shape[1]

print(f"{'rounds':>8}{'s_i.u_ij':>12}{'s_j.u_ij':>12}{'||s_j-s_i||':>14}{'||s_i||':>10}")
print("-" * 56)
for rounds in (0, 2, 4, 8, 12):
    eps = sample_zero_com_noise(
        x0.shape, mask, config.flow.noise_scale, device, x0.dtype,
        torch.Generator().manual_seed(0), coords=x0, smoothing_rounds=rounds,
        knn_k=config.model.graph.knn_k,
    )
    feats = flow_state_edge_features(
        eps.reshape(n, 3), graph.edge_index, graph.relative_vectors, graph.distances
    )
    print(f"{rounds:>8}{feats[:, 0].std():>12.4f}{feats[:, 1].std():>12.4f}"
          f"{feats[:, 2].mean():>14.4f}{eps.reshape(n,3).norm(dim=-1).mean():>10.4f}")
print("\nA collapsing third column means the feature stops carrying information.")
