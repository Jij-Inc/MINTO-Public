#!/usr/bin/env python3
"""
MINTO環境メタデータ自動収集機能のテスト
"""

import pathlib
import tempfile

import pytest

import minto


def test_environment_metadata_collection():
    """環境メタデータの自動収集機能をテスト"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)

        # 環境メタデータ収集を有効にした実験
        exp = minto.Experiment(
            name="test_env_collection", savedir=tmp_path, collect_environment=True
        )

        # 実験実行
        with exp:
            exp.log_global_parameter("test_param", "test_value")

        # 環境情報が収集されていることを確認
        env_info = exp.get_environment_info()
        assert env_info is not None
        assert "os_name" in env_info
        assert "python_version" in env_info
        assert "cpu_count" in env_info
        assert "memory_total" in env_info
        assert "package_versions" in env_info
        assert "timestamp" in env_info


def test_environment_metadata_disabled():
    """環境メタデータ収集無効化をテスト"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)

        # 環境メタデータ収集を無効にした実験
        exp = minto.Experiment(
            name="test_env_disabled", savedir=tmp_path, collect_environment=False
        )

        # 実験実行
        with exp:
            exp.log_global_parameter("test_param", "test_value")

        # 環境情報が収集されていないことを確認
        env_info = exp.get_environment_info()
        assert env_info is None


def test_environment_metadata_persistence():
    """環境メタデータの永続化をテスト"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)

        # 実験作成と実行
        exp = minto.Experiment(
            name="test_env_persistence", savedir=tmp_path, collect_environment=True
        )

        with exp:
            exp.log_global_parameter("test_param", "test_value")

        # 元の環境情報を取得
        original_env_info = exp.get_environment_info()
        assert original_env_info is not None

        # 実験保存
        exp.save()

        # 実験読み込み
        loaded_exp = minto.Experiment.load_from_dir(exp.savedir)
        loaded_env_info = loaded_exp.get_environment_info()

        # 環境情報が保持されていることを確認
        assert loaded_env_info is not None
        assert loaded_env_info["os_name"] == original_env_info["os_name"]
        assert loaded_env_info["python_version"] == original_env_info["python_version"]


def test_environment_metadata_ommx_archive():
    """OMMX アーカイブでの環境メタデータ永続化をテスト"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)

        # 実験作成と実行
        exp = minto.Experiment(
            name="test_env_ommx", savedir=tmp_path, collect_environment=True
        )

        with exp:
            exp.log_global_parameter("test_param", "test_value")

        # 元の環境情報を取得
        original_env_info = exp.get_environment_info()
        assert original_env_info is not None

        # OMMX アーカイブとして保存
        ommx_file = tmp_path / "test_env.ommx"
        exp.save_as_ommx_archive(ommx_file)

        # OMMX アーカイブから読み込み
        loaded_exp = minto.Experiment.load_from_ommx_archive(ommx_file)
        loaded_env_info = loaded_exp.get_environment_info()

        # 環境情報が保持されていることを確認
        assert loaded_env_info is not None
        assert loaded_env_info["os_name"] == original_env_info["os_name"]
        assert loaded_env_info["python_version"] == original_env_info["python_version"]


def test_environment_info_methods():
    """環境情報表示メソッドをテスト"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)

        # 環境メタデータ収集を有効にした実験
        exp = minto.Experiment(
            name="test_env_methods", savedir=tmp_path, collect_environment=True
        )

        with exp:
            exp.log_global_parameter("test_param", "test_value")

        # get_environment_info メソッドのテスト
        env_info = exp.get_environment_info()
        assert env_info is not None
        assert isinstance(env_info, dict)

        # print_environment_summary メソッドのテスト(エラーが発生しないことを確認)
        try:
            exp.print_environment_summary()
        except Exception as e:
            pytest.fail(f"print_environment_summary raised an exception: {e}")


if __name__ == "__main__":
    # 直接実行時のテスト
    test_environment_metadata_collection()
    test_environment_metadata_disabled()
    test_environment_metadata_persistence()
    test_environment_metadata_ommx_archive()
    test_environment_info_methods()
    print("All environment metadata tests passed!")
