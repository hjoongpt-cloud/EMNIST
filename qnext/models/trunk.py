import torch
import torch.nn as nn
from qnext.core.frontend import Frontend
from qnext.core.slots import Slots
from qnext.core.heads import SlotHead
from qnext.core.attn import SlotMHSA, LocalSelfAttn
from qnext.core.aggregators import aggregate_logits

class Trunk(nn.Module):
    def __init__(self, K=150, D=64, enc_act="gelu",
                 wta_mode="soft_top1", wta_tau=0.7, wta_k=1,
                 attn_mode="slot",
                 aggregator="logit_mean",
                 num_classes=47,
                 init_from_ae=None,
                 only_load_conv1=False,
                 only_load_proj=False,
                 # runtime-scheduled knobs
                 slotdrop_p=0.1,
                 jitter_px=1,
                 jitter_alpha=0.1,
                 diversity_enable=False,
                 diversity_lambda=0.0,
                 # 2D PE
                 use_2dpe=False, pe_pairs=16, pe_alpha=0.5,
                 # new: dropouts
                 feature_dropout_p=0.0,
                 head_dropout_p=0.0):
        super().__init__()
        assert attn_mode in ["slot", "local"], "Only 'slot' or 'local' are supported."

        # --- Frontend (K,D는 인자 그대로 전달) ---
        self.enc = Frontend(
            K=K, D=D,
            enc_act=enc_act, wta_mode=wta_mode, wta_tau=wta_tau, wta_k=wta_k,
            use_2dpe=use_2dpe, pe_pairs=pe_pairs, pe_alpha=pe_alpha,
        )
        if init_from_ae is not None and len(str(init_from_ae)) > 0:
            loaded = self.enc.load_from_ae_dir(init_from_ae,
                                               only_conv1=only_load_conv1,
                                               only_proj=only_load_proj)
            print(f"[Trunk] init_from_ae='{init_from_ae}' loaded={loaded}")

        # --- Slots & Head ---
        # H,W는 데이터 해상도/stride에 맞춰 조정 가능(여기선 28x28 가정)
        self.slots = Slots(H=28, W=28, scheme="9C4")
        self.head = SlotHead(d_in=D, d_hid=128, num_classes=num_classes, dropout=0.0)

        self.aggregator = aggregator
        self.attn_mode = attn_mode

        # --- Attn ---
        if attn_mode == "slot":
            self.attn = SlotMHSA(d=128, heads=2, dropout=0.0)
        else:  # "local"
            self.attn = LocalSelfAttn(d=128, kernel_size=3)

        # --- runtime knobs (스케줄이 여기 값을 갱신) ---
        self.slotdrop_p = float(slotdrop_p)
        self.jitter_px = int(jitter_px)
        self.jitter_alpha = float(jitter_alpha)
        self.diversity_enable = bool(diversity_enable)
        self.diversity_lambda = float(diversity_lambda)

        # --- Dropouts ---
        self.feature_dropout_p = float(feature_dropout_p)
        self.head_dropout_p = float(head_dropout_p)
        self.feature_do = nn.Dropout(self.feature_dropout_p) if self.feature_dropout_p > 0 else nn.Identity()
        self.head_do = nn.Dropout(self.head_dropout_p)       if self.head_dropout_p > 0 else nn.Identity()

    # ---- 공용 경로: 슬롯 피처까지 만들고 로짓 & 슬롯피처 반환 ----
    def _forward_core(self, x):
        """
        반환:
          logits: [B,C]
          logits_s: [B,S,C]
          slot_feats: [B,S,Dh]  (penultimate 결과, 예: Dh=128)
          aux: enc의 보조 출력
        """
        z, aux = self.enc(x)  # [B,D,H,W], aux: {"a","a_wta","wta",...}

        # 슬롯 임베딩(여기서 slot-drop & jitter 반영)
        slot_emb = self.slots.embed(
            z,
            jitter_px=self.jitter_px,
            jitter_alpha=self.jitter_alpha,
            slotdrop_p=self.slotdrop_p,
            training=self.training
        )  # [B,S,D]

        # feature dropout (training시에만 의미)
        slot_emb = self.feature_do(slot_emb)

        # penultimate -> attn -> cls
        h = self.head.penultimate(slot_emb)  # [B,S,128]
        h = self.attn(h)                     # [B,S,128] (mode별 어텐션)
        h = self.head_do(h)                  # head dropout

        logits_s = self.head.cls(h)          # [B,S,C]
        logits = aggregate_logits(logits_s, mode=self.aggregator)  # [B,C]
        return logits, logits_s, h, aux

    def forward(self, x, return_aux=False):
        logits, logits_s, h, aux = self._forward_core(x)
        if return_aux:
            return logits, {"slot_logits": logits_s, "slot_emb": h, "enc_aux": aux}
        return logits

    @torch.no_grad()
    def forward_with_feats(self, x):
        """
        학습/평가 루프에서 cos-sim 또는 SupCon에 쓰는 API
        반환: (logits [B,C], slot_feats [B,S,Dh])
        """
        logits, _, h, _ = self._forward_core(x)
        return logits, h
