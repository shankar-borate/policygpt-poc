import tempfile
import unittest
from pathlib import Path

from policygpt.config import Config
from policygpt.models import SourceReference
from policygpt.api.routes.chat import PolicyApiServer
from policygpt.api.renderers.ui import WebUIRenderer


class ReferenceViewerTests(unittest.TestCase):
    def test_web_ui_renderer_versions_static_assets(self) -> None:
        renderer = WebUIRenderer(Path("web").resolve())

        html = renderer.render_index()

        self.assertIn('/static/styles.css?v=', html)
        self.assertIn('/static/usage-widget.css?v=', html)
        self.assertIn('/static/app.js?v=', html)
        self.assertIn('/static/usage-widget.js?v=', html)

    def test_serialize_source_uses_section_aware_viewer_link(self) -> None:
        source = SourceReference(
            document_title="Travel Policy",
            section_title="Eligibility",
            source_path=r"D:\policy-mgmt\data\policies\travel_policy.txt",
            score=0.91,
            section_order_index=2,
        )

        payload = PolicyApiServer.serialize_source(source)

        self.assertEqual(payload["section_order_index"], 2)
        self.assertIn("/api/documents/view?", payload["document_url"])
        self.assertIn("section_index=2", payload["document_url"])
        self.assertIn("section_title=Eligibility", payload["document_url"])

    def test_view_document_highlights_requested_section(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            document_path = Path(temp_dir) / "travel_policy.txt"
            document_path.write_text(
                "\n".join(
                    [
                        "Travel Policy",
                        "",
                        "Eligibility",
                        "Employees with manager approval can claim travel.",
                        "",
                        "Approval Process",
                        "Submit the request in the portal.",
                        "Finance validates the reimbursement.",
                    ]
                ),
                encoding="utf-8",
            )

            server = PolicyApiServer(
                config=Config(document_folder=temp_dir, debug=False),
                web_dir=Path("web").resolve(),
            )

            response = server.view_document(
                path=str(document_path),
                section_index=1,
                section_title="Approval Process",
            )

            html = response.body.decode("utf-8")

            self.assertIn('data-target=', html)
            self.assertIn("section-1", html)
            self.assertIn("Approval Process", html)
            self.assertIn("Open original file", html)
            self.assertIn("/static/document-viewer.css?v=", html)
            self.assertIn("/static/document-viewer.js?v=", html)


if __name__ == "__main__":
    unittest.main()
