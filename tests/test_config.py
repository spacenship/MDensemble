from pathlib import Path

from protein_flow.config import Config, load_config, save_config

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


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
