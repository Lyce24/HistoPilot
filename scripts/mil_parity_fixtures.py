"""Write tests/fixtures/mil_parity.json: golden outputs of the authors' MIL code.

The fixture lets tests/test_mil_parity.py check HistoPilot's CLAM, TransMIL, DSMIL,
DTFD-MIL and ACMIL against the official implementations without that code. This script
runs each official model once, on weights and inputs from tests/support/mil_parity.py,
and records its logits, attention, training losses and gradient norms together with the
name, shape and HistoPilot counterpart of every official tensor.

It needs shallow clones of the official repositories at the commits in SOURCES, under
one folder, and einops and future importable (for example installed with
``uv pip install --target DEPS einops future``):

    python scripts/mil_parity_fixtures.py --refs REFS --deps DEPS          # write
    python scripts/mil_parity_fixtures.py --refs REFS --deps DEPS --check  # compare

Official code is imported from the clones and run on the CPU; nothing is installed.
"""

import argparse
import importlib.util
import io
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "tests" / "fixtures" / "mil_parity.json"
SOURCES = {
    "CLAM": "https://github.com/mahmoodlab/CLAM 53e2409",
    "smooth-topk": "https://github.com/oval-group/smooth-topk 12c1645",
    "TransMIL": "https://github.com/szc19990412/TransMIL 9d6aee5",
    "nystrom-attention": "https://github.com/lucidrains/nystrom-attention fbe9eba",
    "dsmil-wsi": "https://github.com/binli123/dsmil-wsi 80465ed",
    "DTFD-MIL": "https://github.com/hrzhang1123/DTFD-MIL 10964f4",
    "ACMIL": "https://github.com/dazhangyu123/ACMIL e53d19c",
}


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def prefixed(prefixes):
    def mapping(key):
        for old, new in prefixes:
            if key.startswith(old):
                return new + key[len(old) :]
        raise KeyError(key)

    return mapping


def assign(model, mapping, seed, skip=()):
    """Fill the official model's tensors from the shared generator; list them for the test."""
    import torch
    from support.mil_parity import tensor_values

    tensors, state = [], model.state_dict()
    for index, (name, value) in enumerate(
        item for item in state.items() if not item[0].startswith(skip)
    ):
        shape = list(value.shape)
        tensors.append([name, mapping(name), shape])
        state[name] = torch.from_numpy(tensor_values(name, shape, seed, index))
    model.load_state_dict(state)
    return tensors


def norms(named, mapping, grads=None):
    return {
        mapping(name): float((grads[name] if grads else parameter.grad).double().norm())
        for name, parameter in named
        if (grads[name] if grads else parameter.grad) is not None
    }


def floats(tensor):
    return [float(value) for value in tensor.detach().double().reshape(-1)]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refs", required=True, type=Path, help="folder with the clones")
    parser.add_argument("--deps", required=True, type=Path, help="folder with einops, future")
    parser.add_argument("--check", action="store_true", help="exit 1 if the fixture differs")
    args = parser.parse_args()
    refs = args.refs.resolve()
    sys.path[:0] = [
        str(args.deps.resolve()),
        str(ROOT),
        str(ROOT / "tests"),
        str(refs / "smooth-topk"),
        str(refs / "nystrom-attention"),
        str(refs / "DTFD-MIL"),
        str(refs / "ACMIL"),
    ]
    import numpy as np
    import torch
    from support.mil_parity import features, svm_inputs
    from torch import nn
    from torch.nn import functional as F

    torch.set_num_threads(1)
    cases = []

    # --- CLAM --------------------------------------------------------------------------
    from topk.svm import SmoothTop1SVM

    clam = load(refs / "CLAM/models/model_clam.py", "ref_model_clam")

    def clam_mapping(key):
        # Attn_Net (ungated) keeps its layers in `module`: Linear, Tanh, [Dropout], Linear.
        key = key.replace("attention_net.3.module.0.", "attention_a.0.")
        key = key.replace("attention_net.3.module.3.", "attention_c.")
        key = key.replace("attention_net.3.module.2.", "attention_c.")
        key = key.replace("attention_net.0.", "patch_embed.0.").replace("attention_net.3.", "")
        return "classifier." + key[len("classifiers.") :] if key.startswith("classifiers.") else key

    for variant, classes, loss, gated, subtyping, patches, label in (
        ("sb", 2, "svm", True, False, 64, 1),
        ("sb", 3, "svm", True, True, 40, 2),
        ("sb", 2, "ce", False, False, 30, 0),
        ("mb", 3, "svm", True, False, 50, 1),
        ("mb", 3, "ce", True, True, 33, 0),
    ):
        case = {
            "id": f"clam_{variant}-{classes}-{loss}{'-ungated' if not gated else ''}{'-subtyping' if subtyping else ''}",
            "model": f"clam_{variant}",
            "source": "CLAM",
            "featureDim": 24,
            "outputs": classes,
            "patches": patches,
            "inputSeed": len(cases) + 101,
            "weightSeed": len(cases) + 1,
            "label": label,
            "loss": "ce",
            "recipe": {
                "embedDim": 512,
                "dropout": 0.25,
                "clamAttentionDim": 256,
                "clamGatedAttention": gated,
                "clamInstanceSamples": 8,
                "clamBagWeight": 0.7,
                "clamInstanceLoss": loss,
                "clamSubtyping": subtyping,
            },
        }
        kind = clam.CLAM_SB if variant == "sb" else clam.CLAM_MB
        official = kind(
            gate=gated,
            size_arg="small",
            dropout=0.25,
            k_sample=8,
            n_classes=classes,
            instance_loss_fn=SmoothTop1SVM(n_classes=2) if loss == "svm" else nn.CrossEntropyLoss(),
            subtyping=subtyping,
            embed_dim=24,
        ).eval()
        case["tensors"] = assign(
            official, clam_mapping, case["weightSeed"], skip=("instance_loss_fn.",)
        )
        h = torch.from_numpy(features(case))
        target = torch.tensor([label])
        logits, _, y_hat, raw, results = official(h, label=target, instance_eval=True)
        branch = 0 if variant == "sb" else int(y_hat)
        total = 0.7 * F.cross_entropy(logits, target) + 0.3 * results["instance_loss"]
        total.backward()
        case["eval"] = {
            "logits": floats(logits[0]),
            "attention": floats(F.softmax(raw, dim=1)[branch]),
        }
        case["fit"] = {
            "loss": float(total),
            "gradNorms": norms(official.named_parameters(), clam_mapping),
        }
        cases.append(case)

    # --- TransMIL ------------------------------------------------------------------------
    torch.Tensor.cuda = lambda self, *a, **k: self  # the reference moves its class token to CUDA
    transmil = load(refs / "TransMIL/models/TransMIL.py", "ref_transmil")

    def transmil_mapping(key):
        key = key.replace("_fc1.", "patch_embed.").replace("cls_token", "class_token")
        key = key.replace("layer1.", "layers.0.").replace("layer2.", "layers.1.")
        key = key.replace(".attn.res_conv.", ".attention.residual.").replace(
            ".attn.", ".attention."
        )
        return key.replace("pos_layer.", "position.").replace("_fc2.", "classifier.")

    for patches, classes, label in ((5, 2, 1), (300, 2, 0), (1100, 3, 2)):
        case = {
            "id": f"transmil-{patches}-{classes}",
            "model": "transmil",
            "source": "TransMIL + nystrom-attention",
            "featureDim": 1024,
            "outputs": classes,
            "patches": patches,
            "inputSeed": len(cases) + 101,
            "weightSeed": len(cases) + 1,
            "label": label,
            "loss": "ce",
            "recipe": {},
        }
        official = transmil.TransMIL(n_classes=classes).eval()
        case["tensors"] = assign(official, transmil_mapping, case["weightSeed"])
        logits = official(data=torch.from_numpy(features(case))[None])["logits"]
        loss = F.cross_entropy(logits, torch.tensor([label]))
        loss.backward()
        case["eval"] = {"logits": floats(logits[0])}
        case["fit"] = {
            "loss": float(loss),
            "gradNorms": norms(official.named_parameters(), transmil_mapping),
        }
        cases.append(case)

    # --- DSMIL ---------------------------------------------------------------------------
    dsmil = load(refs / "dsmil-wsi/dsmil.py", "ref_dsmil")
    dsmil_mapping = prefixed(
        (
            ("i_classifier.fc.0.", "instance_classifier."),
            ("b_classifier.q.", "query."),
            ("b_classifier.v.", "value."),
            ("b_classifier.fcc.", "classifier."),
        )
    )
    for classes, nonlinear, passing, patches in (
        (1, True, False, 60),
        (3, True, True, 45),
        (1, False, True, 20),
    ):
        case = {
            "id": f"dsmil-{classes}{'-linear' if not nonlinear else ''}{'-value' if passing else ''}",
            "model": "dsmil",
            "source": "dsmil-wsi",
            "featureDim": 32,
            "outputs": classes,
            "patches": patches,
            "inputSeed": len(cases) + 101,
            "weightSeed": len(cases) + 1,
            "label": 1,
            "loss": "bce",
            "recipe": {"dsmilNonlinearQuery": nonlinear, "dsmilPassingValue": passing},
        }
        official = dsmil.MILNet(
            dsmil.FCLayer(32, classes),
            dsmil.BClassifier(32, classes, dropout_v=0.0, nonlinear=nonlinear, passing_v=passing),
        ).eval()
        case["tensors"] = assign(official, dsmil_mapping, case["weightSeed"])
        instance, bag, attention, _ = official(torch.from_numpy(features(case)))
        branch = 0 if classes == 1 else int(bag.argmax())
        case["eval"] = {"logits": floats(bag[0]), "attention": floats(attention[:, branch])}
        if classes == 1:
            # The reference's binary setting: BCE on both streams, 0.5 / 0.5.
            target = torch.tensor([[1.0]])
            criterion = nn.BCEWithLogitsLoss()
            loss = 0.5 * criterion(bag.view(1, -1), target) + 0.5 * criterion(
                instance.max(0).values.view(1, -1), target
            )
            loss.backward()
            case["fit"] = {
                "loss": float(loss),
                "gradNorms": norms(official.named_parameters(), dsmil_mapping),
            }
        cases.append(case)

    # --- DTFD-MIL ------------------------------------------------------------------------
    stub = types.ModuleType("torch.utils.tensorboard")
    stub.SummaryWriter = object
    sys.modules["torch.utils.tensorboard"] = stub
    from Model.Attention import Attention_Gated, Attention_with_Classifier
    from Model.network import Classifier_1fc, DimReduction
    from utils import get_cam_1d

    dtfd_mapping = prefixed(
        (
            ("attCls.attention.attention_V.0.", "tier2_attention.attention_v.0."),
            ("attCls.attention.attention_U.0.", "tier2_attention.attention_u.0."),
            ("attCls.attention.attention_weights.", "tier2_attention.attention_w."),
            ("attCls.classifier.fc.", "classifier.fc."),
            ("attention.attention_V.0.", "tier1_attention.attention_v.0."),
            ("attention.attention_U.0.", "tier1_attention.attention_u.0."),
            ("attention.attention_weights.", "tier1_attention.attention_w."),
            ("classifier.fc.", "tier1_classifier.fc."),
            ("dimReduction.", "patch_embed."),
        )
    )

    def dtfd_slide(m, tfeat, feat_index, numGroup, distill, instance_per_group):
        """The per-slide body of the reference's test_attention_DTFD_preFeat_MultipleMean."""
        classifier, attention, dimReduction, UClassifier = (
            m["classifier"],
            m["attention"],
            m["dimReduction"],
            m["attCls"],
        )
        midFeat = dimReduction(tfeat)
        AA = attention(midFeat, isNorm=False).squeeze(0)
        index_chunk_list = np.array_split(np.array(feat_index), numGroup)
        index_chunk_list = [sst.tolist() for sst in index_chunk_list]
        slide_d_feat, slide_sub_preds = [], []
        for tindex in index_chunk_list:
            idx_tensor = torch.LongTensor(tindex)
            tmidFeat = midFeat.index_select(dim=0, index=idx_tensor)
            tAA = AA.index_select(dim=0, index=idx_tensor)
            tAA = torch.softmax(tAA, dim=0)
            tattFeats = torch.einsum("ns,n->ns", tmidFeat, tAA)
            tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)
            tPredict = classifier(tattFeat_tensor)
            slide_sub_preds.append(tPredict)
            patch_pred_logits = get_cam_1d(classifier, tattFeats.unsqueeze(0)).squeeze(0)
            patch_pred_logits = torch.transpose(patch_pred_logits, 0, 1)
            patch_pred_softmax = torch.softmax(patch_pred_logits, dim=1)
            _, sort_idx = torch.sort(patch_pred_softmax[:, -1], descending=True)
            if distill == "MaxMinS":
                topk_idx_max = sort_idx[:instance_per_group].long()
                topk_idx_min = sort_idx[-instance_per_group:].long()
                topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=0)
                slide_d_feat.append(tmidFeat.index_select(dim=0, index=topk_idx))
            elif distill == "MaxS":
                slide_d_feat.append(
                    tmidFeat.index_select(dim=0, index=sort_idx[:instance_per_group].long())
                )
            elif distill == "AFS":
                slide_d_feat.append(tattFeat_tensor)
        return UClassifier(torch.cat(slide_d_feat, dim=0)), torch.cat(slide_sub_preds, dim=0)

    names = {"afs": "AFS", "maxmins": "MaxMinS", "maxs": "MaxS"}
    for distill, classes, patches, groups, per_group, residual, splits in (
        ("afs", 2, 50, 4, 1, 0, 1),
        ("maxmins", 2, 37, 4, 1, 2, 1),
        ("maxs", 3, 41, 5, 2, 0, 1),
        ("afs", 2, 50, 4, 1, 0, 3),
    ):
        case = {
            "id": f"dtfd-{distill}-{classes}-{groups}x{per_group}-res{residual}-splits{splits}",
            "model": "dtfd",
            "source": "DTFD-MIL",
            "featureDim": 24,
            "outputs": classes,
            "patches": patches,
            "inputSeed": len(cases) + 101,
            "weightSeed": len(cases) + 1,
            "label": classes - 1,
            "loss": "ce",
            "recipe": {
                "embedDim": 64,
                "dtfdPseudoBags": groups,
                "dtfdTestSplits": splits,
                "dtfdDistill": distill,
                "dtfdDistillPatches": per_group,
                "dtfdResidualBlocks": residual,
                "dtfdAttentionDim": 128,
            },
        }
        m = nn.ModuleDict(
            {
                "classifier": Classifier_1fc(64, classes, 0.0),
                "attention": Attention_Gated(64),
                "dimReduction": DimReduction(24, 64, numLayer_Res=residual),
                "attCls": Attention_with_Classifier(L=64, num_cls=classes, droprate=0.0),
            }
        ).eval()
        case["tensors"] = assign(m, dtfd_mapping, case["weightSeed"])
        h = torch.from_numpy(features(case))
        orders = [
            np.random.RandomState(case["inputSeed"] + 7 + split).permutation(patches).tolist()
            for split in range(splits)
        ]
        case["orders"] = orders
        probabilities = []
        for order in orders:
            logits, sub = dtfd_slide(m, h, order, groups, names[distill], per_group)
            probabilities.append(torch.softmax(logits, dim=1))
        case["eval"] = {"probabilities": floats(torch.cat(probabilities).mean(0))}
        if splits == 1:
            case["eval"]["logits"] = floats(logits[0])
            target = torch.tensor([classes - 1])
            ce = nn.CrossEntropyLoss(reduction="none")
            loss0 = ce(sub, target.repeat(len(sub))).mean()
            loss1 = ce(logits, target).mean()
            named = dict(m.named_parameters())
            tier1 = [k for k in named if not k.startswith("attCls")]
            tier2 = [k for k in named if k.startswith("attCls")]
            # Each tier learns from its own loss, as the reference's two optimizers do.
            grads = dict(
                zip(
                    tier1,
                    torch.autograd.grad(loss0, [named[k] for k in tier1], retain_graph=True),
                    strict=True,
                )
            )
            grads.update(
                zip(tier2, torch.autograd.grad(loss1, [named[k] for k in tier2]), strict=True)
            )
            case["fit"] = {
                "loss": float(loss0 + loss1),
                "gradNorms": norms(m.named_parameters(), dtfd_mapping, grads),
            }
        cases.append(case)

    # --- ACMIL ---------------------------------------------------------------------------
    from architecture.transformer import ACMIL_GA

    acmil_mapping = prefixed(
        (
            ("dimreduction.", "patch_embed."),
            ("attention.attention_V.0.", "attention_v.0."),
            ("attention.attention_U.0.", "attention_u.0."),
            ("attention.attention_weights.", "attention_w."),
            ("Slide_classifier.fc.", "classifier.fc."),
            ("classifier.", "branch_classifiers."),
        )
    )
    for branches, classes, patches, masked, drop in (
        (5, 2, 80, 10, 0.6),
        (1, 3, 30, 0, 0.0),
        (3, 2, 55, 10, 0.6),
    ):
        case = {
            "id": f"acmil-{branches}-{classes}-mask{masked}",
            "model": "acmil",
            "source": "ACMIL",
            "featureDim": 24,
            "outputs": classes,
            "patches": patches,
            "inputSeed": len(cases) + 101,
            "weightSeed": len(cases) + 1,
            "label": classes - 1,
            "loss": "ce",
            "recipe": {
                "embedDim": 64,
                "acmilBranches": branches,
                "acmilMaskedPatches": masked,
                "acmilMaskDrop": drop,
                "acmilAttentionDim": 128,
            },
        }
        conf = types.SimpleNamespace(D_feat=24, D_inner=64, n_class=classes, n_token=branches)
        official = ACMIL_GA(
            conf, D=128, droprate=0, n_token=branches, n_masked_patch=masked, mask_drop=drop
        ).eval()
        case["tensors"] = assign(official, acmil_mapping, case["weightSeed"])
        sub, slide, attention = official(torch.from_numpy(features(case))[None])
        target = torch.tensor([classes - 1])
        criterion = nn.CrossEntropyLoss()
        loss = criterion(slide, target)
        if branches > 1:
            loss = loss + criterion(sub, target.repeat_interleave(branches))
            weights = torch.softmax(attention, dim=-1)
            for i in range(branches):
                for j in range(i + 1, branches):
                    loss = loss + torch.cosine_similarity(
                        weights[:, i], weights[:, j], dim=-1
                    ).mean() / (branches * (branches - 1) / 2)
        loss.backward()
        case["eval"] = {
            "logits": floats(slide[0]),
            "attention": floats(torch.softmax(attention[0], -1).mean(0)),
        }
        case["fit"] = {
            "loss": float(loss),
            "gradNorms": norms(official.named_parameters(), acmil_mapping),
        }
        cases.append(case)

    # --- CLAM's smooth top-1 SVM instance loss -------------------------------------------
    svm_case = {
        "id": "smooth-top1-svm",
        "inputSeed": 7,
        "rows": 64,
        "scales": [0.1, 1.0, 5.0, 50.0],
    }
    logits, targets = svm_inputs(svm_case)
    x = torch.from_numpy(logits).requires_grad_()
    loss = SmoothTop1SVM(n_classes=2)(x, torch.from_numpy(targets))
    loss.backward()
    svm_case.update(loss=float(loss), gradient=floats(x.grad))

    fixture = {
        "version": 1,
        "generator": "scripts/mil_parity_fixtures.py",
        "torch": torch.__version__.split("+")[0],
        "sources": SOURCES,
        "cases": cases,
        "svm": svm_case,
    }
    rendered = json.dumps(fixture, indent=1, sort_keys=True) + "\n"
    if args.check:
        current = TARGET.read_text(encoding="utf-8") if TARGET.exists() else ""
        if current != rendered:
            print(f"{TARGET.relative_to(ROOT)} is out of date.")
            return 1
        return 0
    TARGET.write_text(rendered, encoding="utf-8")
    print(f"Wrote {TARGET.relative_to(ROOT)} ({len(cases)} cases, {len(rendered) // 1024} KiB).")
    return 0


if __name__ == "__main__":
    # SmoothTop1SVM prints when it sets its temperature.
    sys.stdout, real = io.StringIO(), sys.stdout
    try:
        code = main()
    finally:
        captured, sys.stdout = sys.stdout.getvalue(), real
    print("\n".join(line for line in captured.splitlines() if not line.startswith("Setting tau")))
    raise SystemExit(code)
