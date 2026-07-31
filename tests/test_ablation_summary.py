from scripts.summarize_gap_ablation import parse_log, summarize_run


def test_ablation_log_parser_keeps_validation_and_endpoint_at_same_step(tmp_path):
    log = tmp_path / "gap1.log"
    log.write_text(
        "step 2000 | val_loss=12.5 | fm=12.4, bond=0.1 | zero_fm=13.0, "
        "fm_improvement=4.62% | lr=3.000e-04 | fm_by_temp: 320K=3.0\n"
        "step 2000 | endpoint_rmsd: source=2.8, generated=2.7, "
        "improvement=3.57%, win_rate=75.00% | endpoint_physics: "
        "bond=0.1, angle=0.02, clash=0.003\n"
    )
    records = parse_log(log)
    summary = summarize_run(1, records)

    assert records[0]["step"] == 2000
    assert records[0]["endpoint_improvement"] == 3.57
    assert records[0]["endpoint_bond"] == 0.1
    assert summary["best_validation"]["step"] == summary["best_endpoint"]["step"]
