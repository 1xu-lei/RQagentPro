"""
企业信息接口
"""

import os
from pathlib import Path
from fastapi import APIRouter
from fastapi.responses import FileResponse

router = APIRouter(prefix="/api/v1/company")

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"


# ---- docx 转 HTML ----

def docx_to_html(file_path: Path) -> str:
    """将 docx 文档转换为带格式的 HTML"""
    try:
        from docx import Document
        return _docx_to_html_via_python_docx(file_path, Document)
    except ImportError:
        return _docx_to_html_via_zipfile(file_path)


def _docx_to_html_via_python_docx(file_path: Path, Document) -> str:
    """使用 python-docx 解析"""
    import re

    doc = Document(str(file_path))
    html_parts = []

    for para in doc.paragraphs:
        style_name = para.style.name if para.style else "Normal"
        if style_name.startswith("Heading"):
            level = re.sub(r"\D", "", style_name) or "1"
            tag = f"h{level}"
        else:
            tag = "p"

        align = ""
        if para.alignment:
            align_map = {0: "left", 1: "center", 2: "right", 3: "justify"}
            align = f' style="text-align:{align_map.get(para.alignment, "left")}"'

        text_parts = []
        for run in para.runs:
            text = run.text
            if not text:
                continue
            if run.bold:
                text = f"<strong>{text}</strong>"
            if run.italic:
                text = f"<em>{text}</em>"
            if run.underline:
                text = f"<u>{text}</u>"
            text_parts.append(text)

        text_content = "".join(text_parts) if text_parts else para.text
        if not text_content.strip():
            text_content = "&nbsp;"

        html_parts.append(f"<{tag}{align}>{text_content}</{tag}>")

    return _wrap_html("".join(html_parts))


def _docx_to_html_via_zipfile(file_path: Path) -> str:
    """fallback: 使用 zipfile + xml 解析"""
    import zipfile
    from xml.etree import ElementTree as ET

    NS_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

    html_parts = []
    with zipfile.ZipFile(str(file_path), "r") as z:
        xml_content = z.read("word/document.xml")
        root = ET.fromstring(xml_content)

        for para in root.iter(f"{{{NS_W}}}p"):
            style = ""
            pPr = para.find(f"{{{NS_W}}}pPr")
            if pPr is not None:
                pStyle = pPr.find(f"{{{NS_W}}}pStyle")
                if pStyle is not None:
                    style = pStyle.get(f"{{{NS_W}}}val", "")

            if style and style.startswith("Heading"):
                level = "".join(c for c in style if c.isdigit()) or "1"
                tag = f"h{level}"
            else:
                tag = "p"

            text_parts = []
            for run in para.iter(f"{{{NS_W}}}r"):
                run_text = []
                for t in run.iter(f"{{{NS_W}}}t"):
                    run_text.append(t.text or "")
                text = "".join(run_text)
                if not text:
                    continue
                bold = run.find(f"{{{NS_W}}}rPr/{{{NS_W}}}b") is not None
                if bold:
                    text = f"<strong>{text}</strong>"
                text_parts.append(text)

            text_content = "".join(text_parts)
            if not text_content.strip():
                text_content = "&nbsp;"

            html_parts.append(f"<{tag}>{text_content}</{tag}>")

    return _wrap_html("".join(html_parts))


def _wrap_html(body: str) -> str:
    """包裹统一样式"""
    style = """
    <style>
        .docx-content { font-size:14px; line-height:1.8; color:#2c3038; }
        .docx-content h1 { font-size:22px; font-weight:700; margin:16px 0 8px; color:#1a6fb5; }
        .docx-content h2 { font-size:18px; font-weight:600; margin:14px 0 6px; color:#1a6fb5; }
        .docx-content h3 { font-size:16px; font-weight:600; margin:12px 0 6px; color:#1a5f9e; }
        .docx-content p { margin:6px 0; }
        .docx-content strong { color:#1a5f9e; }
    </style>
    <div class="docx-content">
    """
    return style + body + "</div>"


# ---- 路由 ----

@router.get("/profile")
async def get_company_profile():
    """企业概况"""
    file_path = DATA_DIR / "公司概述1.docx"
    try:
        html = docx_to_html(file_path)
        return {"status": "ok", "content": html, "format": "html"}
    except FileNotFoundError:
        return {"status": "error", "content": "企业概况文件不存在"}
    except Exception as e:
        return {"status": "error", "content": f"读取文件失败: {str(e)}"}


@router.get("/culture")
async def get_company_culture():
    """企业文化"""
    file_path = DATA_DIR / "企业文化1.docx"
    try:
        html = docx_to_html(file_path)
        return {"status": "ok", "content": html, "format": "html"}
    except FileNotFoundError:
        return {"status": "error", "content": "企业文化文件不存在"}
    except Exception as e:
        return {"status": "error", "content": f"读取文件失败: {str(e)}"}


@router.get("/qrcode")
async def get_company_qrcode():
    """企业二维码"""
    file_path = DATA_DIR / "企业二维码.png"
    if file_path.exists():
        return FileResponse(str(file_path), media_type="image/png")
    return {"status": "error", "message": "二维码文件不存在"}


@router.get("/background")
async def get_company_background():
    """背景图"""
    file_path = DATA_DIR / "背景图.png"
    if file_path.exists():
        return FileResponse(str(file_path), media_type="image/png")
    return {"status": "error", "message": "背景图文件不存在"}
