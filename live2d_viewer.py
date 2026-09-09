from __future__ import annotations

import json
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit, quote

from PySide6.QtCore import QUrl, Qt
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

from live2d_models import Live2DModel

try:
    from PySide6.QtWebEngineWidgets import QWebEngineView

    WEBENGINE_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised on minimal installations
    QWebEngineView = None  # type: ignore[assignment,misc]
    WEBENGINE_AVAILABLE = False


class _AssetHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: object, asset_root: Path, model_root: Path, **kwargs: object) -> None:
        self.asset_root = asset_root.resolve()
        self.model_root = model_root.resolve()
        super().__init__(*args, **kwargs)

    def translate_path(self, path: str) -> str:
        requested = unquote(urlsplit(path).path)
        root = self.model_root if requested.startswith("/__model__/") else self.asset_root
        relative = requested.removeprefix("/__model__/") if root == self.model_root else requested.lstrip("/")
        candidate = (root / relative).resolve()
        if candidate != root and root not in candidate.parents:
            return str(root / "__missing__")
        return str(candidate)

    def log_message(self, _format: str, *_args: object) -> None:
        return


class _AssetServer:
    def __init__(self, asset_root: Path, model_root: Path) -> None:
        handler = partial(
            _AssetHandler,
            asset_root=asset_root,
            model_root=model_root,
        )
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, name="live2d-assets", daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


class Live2DView(QFrame):
    """Embedded Hiyori/Mira renderer with a QLabel fallback."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("live2dView")
        self._server: _AssetServer | None = None
        self._model: Live2DModel | None = None
        self._web: QWebEngineView | None = None
        self._fallback = QLabel("Live2D 预览需要 PySide6 WebEngine\n请运行 setup.ps1 安装完整依赖")
        self._fallback.setObjectName("live2dFallback")
        self._fallback.setAlignment(Qt.AlignCenter)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        if WEBENGINE_AVAILABLE:
            self._web = QWebEngineView(self)
            self._web.setContextMenuPolicy(Qt.NoContextMenu)
            layout.addWidget(self._web)
        else:
            layout.addWidget(self._fallback)

    @property
    def available(self) -> bool:
        return self._web is not None

    def set_model(self, model: Live2DModel | None) -> None:
        self._model = model
        if not model:
            if self._web:
                self._web.setHtml("<body style='background:transparent'></body>")
            return
        if not self._web:
            self._fallback.setText(f"{model.name}\nLive2D WebEngine 未安装")
            return
        if self._server:
            self._server.close()
        asset_root = Path(__file__).resolve().parent / "assets" / "live2d"
        self._server = _AssetServer(asset_root, model.root)
        model_url = f"{self._server.url}/live2d-viewer.html?model={quote('/__model__/' + model.relative_manifest)}"
        self._web.setUrl(QUrl(model_url))

    def set_state(self, state: str) -> None:
        if not self._web:
            return
        script = f"window.setState && window.setState({json.dumps(state)});"
        self._web.page().runJavaScript(script)

    def closeEvent(self, event) -> None:  # noqa: ANN001
        if self._server:
            self._server.close()
            self._server = None
        super().closeEvent(event)
