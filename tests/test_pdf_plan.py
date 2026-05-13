from pathlib import Path

from pypdf import PdfWriter

from heta.kb import paths
from heta.kb.pdf_plan import PDF_PAGE_THRESHOLD, estimate_pdf_pages, plan_insert_files


def test_plan_insert_files_splits_large_pdf_and_keeps_original(tmp_path: Path) -> None:
    source = tmp_path / "large.pdf"
    _write_pdf(source, PDF_PAGE_THRESHOLD + 1)

    prepared, plans = plan_insert_files([source], base_dir=tmp_path)

    assert len(plans) == 1
    assert plans[0].enabled is True
    assert plans[0].page_count == PDF_PAGE_THRESHOLD + 1
    assert plans[0].parts == 3
    assert len(prepared) == 3
    assert (paths.raw_dir(tmp_path) / "originals").exists()
    assert all(item.archived_path.exists() for item in prepared)
    assert [estimate_pdf_pages(item.archived_path) for item in prepared] == [40, 40, 1]
    assert all(item.original_path is not None for item in prepared)


def test_plan_insert_files_can_disable_pdf_planning(tmp_path: Path) -> None:
    source = tmp_path / "large.pdf"
    _write_pdf(source, PDF_PAGE_THRESHOLD + 1)

    prepared, plans = plan_insert_files([source], enable_pdf_planning=False, base_dir=tmp_path)

    assert len(prepared) == 1
    assert len(plans) == 1
    assert plans[0].enabled is False
    assert prepared[0].archived_path.exists()
    assert estimate_pdf_pages(prepared[0].archived_path) == PDF_PAGE_THRESHOLD + 1
    assert not (paths.raw_dir(tmp_path) / "originals").exists()


def _write_pdf(path: Path, pages: int) -> None:
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=72, height=72)
    with path.open("wb") as file:
        writer.write(file)
