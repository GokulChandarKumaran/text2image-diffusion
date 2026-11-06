"""Sanity-check the environment: PyTorch, CUDA, Diffusers, Transformers, CLIP."""
import sys


def main() -> None:
    print(f"Python       : {sys.version.split()[0]}")
    try:
        import torch
    except ImportError:
        sys.exit("PyTorch missing -> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128")

    print(f"PyTorch      : {torch.__version__}")
    print(f"CUDA avail.  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"GPU          : {p.name}  ({p.total_memory / 1024**3:.1f} GB, sm_{p.major}{p.minor})")
        x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
        _ = x @ x  # fails here if the wheel lacks kernels for this GPU arch
        print("CUDA matmul  : OK")

    import diffusers, transformers
    print(f"Diffusers    : {diffusers.__version__}")
    print(f"Transformers : {transformers.__version__}")

    from transformers import CLIPModel, CLIPTokenizer
    name = "openai/clip-vit-large-patch14"
    tok = CLIPTokenizer.from_pretrained(name)
    model = CLIPModel.from_pretrained(name)
    ids = tok(["a dog riding a skateboard"], padding="max_length", max_length=77, return_tensors="pt")
    emb = model.get_text_features(**ids)
    print(f"CLIP text emb: shape={tuple(emb.shape)}  -> environment OK")


if __name__ == "__main__":
    main()
