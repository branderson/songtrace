from karaoke_maker.cli import main

def test_version(capsys):
    assert main(["--version"]) == 0
    assert "karaoke-maker" in capsys.readouterr().out
