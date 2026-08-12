from pathlib import Path

from protein_flow.config import Config, load_config, save_config

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
ABLATION_DIR = CONFIG_PATH.parent / "ablations"


def test_load_default_config():
    config = load_config(CONFIG_PATH)
    assert isinstance(config, Config)
    assert config.data.plm_dim == 320
    assert config.model.graph.knn_k == 16
    assert config.loss.physics.apply_to == "euler_step"


def test_save_and_reload_roundtrip(tmp_path):
    config = load_config(CONFIG_PATH)
    config.data.batch_size = 4
    out_path = tmp_path / "roundtrip.yaml"
    save_config(config, out_path)
    reloaded = load_config(out_path)
    assert reloaded.data.batch_size == 4
    assert reloaded.model.decoder.remove_com_velocity is True


def test_config_inherits_and_deep_merges(tmp_path):
    base = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"
    base.write_text("data:\n  batch_size: 7\n  mdcath_frame_gap: 5\ntrain:\n  seed: 11\n")
    child.write_text("base_config: base.yaml\ndata:\n  mdcath_frame_gap: 2\n")

    config = load_config(child)
    assert config.data.batch_size == 7
    assert config.data.mdcath_frame_gap == 2
    assert config.train.seed == 11


def test_config_rejects_circular_inheritance(tmp_path):
    import pytest

    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("base_config: second.yaml\n")
    second.write_text("base_config: first.yaml\n")

    with pytest.raises(ValueError, match="Circular base_config"):
        load_config(first)


def test_mdcath_gap_ablation_configs_are_paired():
    for gap in (1, 2, 5):
        config = load_config(ABLATION_DIR / f"mdcath_gap{gap}.yaml")
        assert config.data.mdcath_frame_gap == gap
        assert config.data.mdcath_sampling_max_frame_gap == 5
        assert config.train.ckpt_dir == f"checkpoints_mdcath_gap{gap}"


def test_dist_timeout_must_outlast_download_timeout():
    """The chunk barrier has to survive a rank sitting in ShardPool.ensure().

    Regression: with the NCCL timeout at 30 min and the downloader's at 180,
    a rotation where one rank's slice was already resident and the other's
    was not aborted the run at exactly 1800 s -- the downloader watchdog that
    would have recovered it could never fire first.
    """
    import pytest

    from protein_flow.config import Config, validate_config

    config = Config()
    config.data.rotation.enabled = True
    config.data.rotation.download_timeout_minutes = 180
    config.train.dist_timeout_minutes = 30

    with pytest.raises(ValueError, match="must exceed"):
        validate_config(config)

    config.train.dist_timeout_minutes = 181
    validate_config(config)  # strictly greater is enough


def test_shipped_rotating_configs_satisfy_the_timeout_ordering():
    for name in (
        "mdcath_backbone_rotate.yaml",
        "mdcath_backbone_rotate_displacement.yaml",
        "mdcath_backbone_rotate_displacement_norigid.yaml",
    ):
        config = load_config(CONFIG_PATH.parent / name)
        assert config.data.rotation.enabled
        assert config.train.dist_timeout_minutes > config.data.rotation.download_timeout_minutes
