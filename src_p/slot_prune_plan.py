# --- slot_prune_plan.py (복붙용 스니펫) --------------------------------------
import json, math
import torch
from p_spmask import build_9c4_masks_14x14, repeat_masks_with_groups

@torch.no_grad()
def eval_slotwise_accuracy(model, loader, device, repeat=3, tau=1.2, min_keep=1, smooth=1e-3):
    """
    각 위치 그룹(126개)마다 repeat개 슬롯(기본=3)을 두었을 때,
    '해당 슬롯만 사용'했을 때의 단독 정확도(slot-only)를 측정.
    -> 그룹 내 softmax 가중치(성능 비례)와 prune 후보 산출.

    리턴:
      plan: {
        "repeat": 3,
        "groups": [
          {"group_id": g, "slot_acc": [a1,a2,a3], "weights": [w1,w2,w3], "keep": [True,False,True]},
          ...
        ],
        "meta": {"min_keep": min_keep, "smooth": smooth, "tau": tau}
      }
    """
    masks, _ = build_9c4_masks_14x14(device=device, dtype=torch.float32)
    masks_r, group_id, local_rank, _ = repeat_masks_with_groups(masks, combos=None, repeat=repeat)
    G = masks.shape[0]        # 126
    M = masks_r.shape[0]      # 126*repeat
    # 슬롯별 정답/총계
    slot_correct = torch.zeros(M, dtype=torch.long, device=device)
    slot_total   = torch.zeros(M, dtype=torch.long, device=device)

    model.eval()
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        tokens, _ = model.trunk(images)
        A_logits, cls_logits = model.slot_head_raw(tokens)  # (B,M,H,W), (B,M,C)

        # 공간 softmax + 마스크 + 재정규화
        B, M_chk, H, W = A_logits.shape
        assert M_chk == M, f"num_slots mismatch: {M_chk} vs {M}"
        A = torch.softmax(A_logits.view(B,M,-1), dim=-1).view(B,M,H,W)
        A_masked = A * masks_r.unsqueeze(0)
        mass = A_masked.sum(dim=(2,3))                       # (B,M)

        # '해당 슬롯만 사용' → 그 슬롯 P=1, 나머지 0
        # 최종 로짓은 그 슬롯의 cls_logits로 대체
        pred_slot = cls_logits.argmax(dim=2)                 # (B,M)
        # 정확도 업데이트
        matches = (pred_slot == labels.unsqueeze(1))         # (B,M)
        slot_correct += matches.sum(dim=0).to(slot_correct.dtype)
        slot_total   += torch.tensor([B]*M, device=device, dtype=torch.long)

    slot_acc = (slot_correct.float() / slot_total.clamp_min(1)).cpu()  # (M,)
    # 그룹별로 정리
    plan = {"repeat": repeat, "groups": [], "meta": {"min_keep": int(min_keep), "smooth": float(smooth), "tau": float(tau)}}
    for g in range(G):
        idx = torch.where(group_id == g)[0]  # length=repeat
        accs = slot_acc[idx]
        # 안정화를 위해 softmax 가중치 (logits = acc/smooth)
        logits = (accs / max(1e-8, smooth)).tolist()
        w = torch.softmax(torch.tensor(logits), dim=0).tolist()
        # prune 규칙: 최소 min_keep은 유지, 그 외는 acc가 그룹 내 중앙값보다 작은 슬롯 drop 등 다양한 규칙 가능
        keep = [True]*len(idx)
        # 예) min_keep만 유지하고 나머지는 acc 순으로 drop 하려면:
        order = torch.argsort(accs, descending=True).tolist()
        for j in range(len(idx)):
            keep[idx.tolist().index(idx[order[j]].item())] = (j < min_keep) or (accs[order[j]] > accs.median())
        plan["groups"].append({
            "group_id": g,
            "slot_acc": [float(a) for a in accs],
            "weights": [float(x) for x in w],
            "keep": keep,
        })
    return plan

def save_plan(plan, path):
    with open(path, "w") as f:
        json.dump(plan, f, indent=2)

def apply_prune_to_ckpt(ckpt, plan):
    """
    ckpt의 slot query와 classifier 등 (M, ...) 축을 prune.
    ckpt 포맷에 맞게 키 이름을 조정해야 합니다.
    - 예시 키: "slot_queries.weight" : (M, D)
             "classifier.weight"     : (C, D)  # 슬롯별이 아니라면 그대로
             "slot_classifier.weight": (M, C, D) or (M, C)
    아래는 (M, D) 쿼리 + (M, C) 슬롯별 분류기 형태 예시.
    """
    M_old = ckpt["slot_queries.weight"].shape[0]
    keep_mask = []
    for g in plan["groups"]:
        repeat = len(g["keep"])
        for r in range(repeat):
            keep_mask.append(g["keep"][r])
    keep_mask = torch.tensor(keep_mask, dtype=torch.bool)
    assert keep_mask.numel() == M_old, f"Mismatch: {keep_mask.numel()} vs {M_old}"

    def take(t, dim=0):
        idx = keep_mask.nonzero(as_tuple=False).squeeze(1).to(t.device)
        return t.index_select(dim, idx)

    ckpt["slot_queries.weight"] = take(ckpt["slot_queries.weight"], dim=0)  # (M_keep, D)
    if "slot_classifier.weight" in ckpt:
        ckpt["slot_classifier.weight"] = take(ckpt["slot_classifier.weight"], dim=0)  # (M_keep, C)
    if "slot_classifier.bias" in ckpt:
        ckpt["slot_classifier.bias"] = take(ckpt["slot_classifier.bias"], dim=0)
    ckpt["meta.num_slots"] = int(keep_mask.sum().item())
    return ckpt
# ---------------------------------------------------------------------------
