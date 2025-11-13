from click.testing import CliRunner

from uhop.cli import main


def test_config_list_runs():
    runner = CliRunner()
    result = runner.invoke(main, ["config", "list"])
    assert result.exit_code == 0, result.output
