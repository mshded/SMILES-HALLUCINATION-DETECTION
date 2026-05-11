"""
aggregation.py — Token aggregation strategy and feature extraction
               (student-implemented).

Converts per-token, per-layer hidden states from the extraction loop in
``solution.py`` into flat feature vectors for the probe classifier.

Two stages can be customised independently:

  1. ``aggregate`` — select layers and token positions, pool into a vector.
  2. ``extract_geometric_features`` — optional hand-crafted features
     (enabled by setting ``USE_GEOMETRIC = True`` in ``solution.py``).

Both stages are combined by ``aggregation_and_feature_extraction``, the
single entry point called from the notebook.
"""

from __future__ import annotations

import torch


TAIL_TOKENS = 32


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Convert per-token hidden states into a single feature vector.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
                        Layer index 0 is the token embedding; index -1 is the
                        final transformer layer.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D feature tensor of shape ``(4 * hidden_dim,)``.

    Final strategy:
        Concatenate four compact representations:
          1. final layer, last real token;
          2. final layer, mean pooling over the last 32 real tokens;
          3. layer -4, last real token;
          4. layer -8, last real token.
    """
    real_positions = attention_mask.nonzero(as_tuple=False)
    last_pos = int(real_positions[-1].item())

    tail_start = max(0, last_pos - TAIL_TOKENS + 1)

    final_layer = hidden_states[-1]
    layer_minus_4 = hidden_states[-4]
    layer_minus_8 = hidden_states[-8]

    final_last = final_layer[last_pos]
    final_tail_mean = final_layer[tail_start : last_pos + 1].mean(dim=0)
    layer_minus_4_last = layer_minus_4[last_pos]
    layer_minus_8_last = layer_minus_8[last_pos]

    feature = torch.cat(
        [
            final_last,
            final_tail_mean,
            layer_minus_4_last,
            layer_minus_8_last,
        ],
        dim=0,
    )

    return feature


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Extract hand-crafted geometric / statistical features from hidden states.

    Called only when ``USE_GEOMETRIC = True`` in ``solution.py``. The final
    best-so-far solution uses ``USE_GEOMETRIC = False``, so this function keeps
    the original no-extra-features behavior.
    """
    return torch.zeros(0, dtype=hidden_states.dtype, device=hidden_states.device)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """Aggregate hidden states and optionally append geometric features.

    Main entry point called from ``solution.py`` for each sample.
    Concatenates the output of ``aggregate`` with that of
    ``extract_geometric_features`` when ``use_geometric=True``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``
                        for a single sample.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.
        use_geometric:  Whether to append geometric features. Controlled by
                        the ``USE_GEOMETRIC`` flag in ``solution.py``.

    Returns:
        A 1-D float tensor of shape ``(feature_dim,)``.
    """
    agg_features = aggregate(hidden_states, attention_mask)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features