from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_demo_mode_renders_result_and_download():
    app_path = Path(__file__).parents[1] / "streamlit_app.py"
    app = AppTest.from_file(app_path, default_timeout=10)
    app.run()
    assert not app.exception

    app.button[0].click().run()
    assert not app.exception
    assert any("Саммари" in subheader.value for subheader in app.subheader)
    assert len(app.download_button) == 1
    assert app.download_button[0].label == "Скачать DOCX"

    app.radio[0].set_value("Локальная обработка аудио").run()
    assert not app.exception
    assert not any("Саммари" in subheader.value for subheader in app.subheader)
    assert len(app.download_button) == 0
