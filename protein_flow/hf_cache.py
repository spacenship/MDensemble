"""Load static HuggingFace assets from the local cache first.

Every ``from_pretrained`` call revalidates its cached files with a HEAD
request to huggingface.co under huggingface_hub's 10s read timeout. On a box
with intermittent egress those HEADs are the dominant failure mode: the ESM2
release is immutable, so the revalidation can only ever confirm what is
already on disk, yet a stalled one blocks whichever dataloader worker issued
it -- and with rank 0 blocked the other ranks sit spinning in their NCCL
wait. Observed as ~16 minutes of no step logs at 100% GPU on rank 1.

``load_cached`` asks for the cached copy first and only touches the network
when the asset is genuinely absent, so a fresh machine still works and pays
the download exactly once.

This deliberately does not use ``HF_HUB_OFFLINE``: the mdCATH shard rotation
(:mod:`protein_flow.data.shard_rotation`) has to stay online, so offline mode
has to be scoped to these static assets rather than to the process.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


def load_cached(loader: Callable[..., Any], model_name: str, **kwargs: Any) -> Any:
    """``loader.from_pretrained(model_name, ...)``, preferring the local cache.

    Args:
        loader: an ``AutoModel``/``AutoTokenizer``-style class.
        model_name: HuggingFace model id.
        **kwargs: forwarded to ``from_pretrained`` unchanged.
    """
    try:
        return loader.from_pretrained(model_name, local_files_only=True, **kwargs)
    except OSError:
        # Missing from the cache, or only partially there. Fall back to the
        # network so a first run still works; later runs hit the cache.
        logger.info(
            "%s is not in the local HuggingFace cache; downloading it once.", model_name
        )
        return loader.from_pretrained(model_name, **kwargs)
