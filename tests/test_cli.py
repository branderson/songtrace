from songtrace.cli import main

def test_version(capsys):
    assert main(["--version"]) == 0
    assert "songtrace" in capsys.readouterr().out
