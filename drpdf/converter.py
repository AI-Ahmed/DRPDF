import re
import math
import logging
import unicodedata
from enum import Enum
import concurrent.futures
from string import Template
from typing import Dict, Literal

import arabic_reshaper
import numpy as np
import bidi.algorithm
from pdfminer.converter import PDFConverter
from pdfminer.layout import LTChar, LTFigure, LTLine, LTPage
from pdfminer.pdffont import PDFCIDFont, PDFUnicodeNotDefined
from pdfminer.pdfinterp import PDFGraphicState, PDFResourceManager
from pdfminer.utils import apply_matrix_pt, mult_matrix
from pymupdf import Font
from tenacity import retry, wait_fixed

from drpdf.translator import (
    AnythingLLMTranslator,
    ArgosTranslator,
    AzureOpenAITranslator,
    AzureTranslator,
    BaseTranslator,
    BingTranslator,
    DeepLTranslator,
    DeepLXTranslator,
    DeepseekTranslator,
    DifyTranslator,
    GeminiTranslator,
    GoogleTranslator,
    GrokTranslator,
    GroqTranslator,
    ModelScopeTranslator,
    OllamaTranslator,
    OpenAIlikedTranslator,
    OpenAITranslator,
    QwenMtTranslator,
    SiliconTranslator,
    TencentTranslator,
    XinferenceTranslator,
    ZhipuTranslator,
    OpenRouterTranslator
)

log = logging.getLogger(__name__)


class PDFConverterEx(PDFConverter):
    def __init__(
        self,
        rsrcmgr: PDFResourceManager,
    ) -> None:
        PDFConverter.__init__(self, rsrcmgr, None, "utf-8", 1, None)

    def begin_page(self, page, ctm) -> None:
        # Override to replace cropbox
        (x0, y0, x1, y1) = page.cropbox
        (x0, y0) = apply_matrix_pt(ctm, (x0, y0))
        (x1, y1) = apply_matrix_pt(ctm, (x1, y1))
        mediabox = (0, 0, abs(x0 - x1), abs(y0 - y1))
        self.cur_item = LTPage(page.pageno, mediabox)

    def end_page(self, page):
        # Override to return command stream
        return self.receive_layout(self.cur_item)

    def begin_figure(self, name, bbox, matrix) -> None:
        # Override to set pageid
        self._stack.append(self.cur_item)
        self.cur_item = LTFigure(name, bbox, mult_matrix(matrix, self.ctm))
        self.cur_item.pageid = self._stack[-1].pageid

    def end_figure(self, _: str) -> None:
        # Override to return command stream
        fig = self.cur_item
        assert isinstance(self.cur_item, LTFigure), str(type(self.cur_item))
        self.cur_item = self._stack.pop()
        self.cur_item.add(fig)
        return self.receive_layout(fig)

    def render_char(
        self,
        matrix,
        font,
        fontsize: float,
        scaling: float,
        rise: float,
        cid: int,
        ncs,
        graphicstate: PDFGraphicState,
    ) -> float:
        # Override to set cid and font
        try:
            text = font.to_unichr(cid)
            assert isinstance(text, str), str(type(text))
        except PDFUnicodeNotDefined:
            text = self.handle_undefined_char(font, cid)
        textwidth = font.char_width(cid)
        textdisp = font.char_disp(cid)
        item = LTChar(
            matrix,
            font,
            fontsize,
            scaling,
            rise,
            text,
            textwidth,
            textdisp,
            ncs,
            graphicstate,
        )
        self.cur_item.add(item)
        item.cid = cid  # hack: insert original character encoding
        item.font = font  # hack: insert original character font
        return item.adv


class Paragraph:
    def __init__(self, y, x, x0, x1, y0, y1, size, brk):
        self.y: float = y  # Initial vertical coordinate
        self.x: float = x  # Initial horizontal coordinate
        self.x0: float = x0  # Left boundary
        self.x1: float = x1  # Right boundary
        self.y0: float = y0  # Top boundary
        self.y1: float = y1  # Bottom boundary
        self.size: float = size  # Font size
        self.brk: bool = brk  # Line break marker


# fmt: off
class TranslateConverter(PDFConverterEx):
    def __init__(
        self,
        rsrcmgr,
        vfont: str = None,
        vchar: str = None,
        thread: int = 0,
        layout={},
        lang_in: str = "",
        lang_out: str = "",
        service: str = "",
        noto_name: str = "",
        noto: Font = None,
        envs: Dict = None,
        prompt: Template = None,
        ignore_cache: bool = False,
        column_type: Literal["one", "two"] = "one"
    ) -> None:
        super().__init__(rsrcmgr)
        self.vfont = vfont
        self.vchar = vchar
        self.thread = thread
        self.layout = layout
        self.noto_name = noto_name
        self.noto = noto
        self.translator: BaseTranslator = None
        self.column_type = column_type
        # e.g. "ollama:gemma2:9b" -> ["ollama", "gemma2:9b"]
        param = service.split(":", 1)
        service_name = param[0]
        service_model = param[1] if len(param) > 1 else None
        if not envs:
            envs = {}
        for translator in [GoogleTranslator, BingTranslator, DeepLTranslator, DeepLXTranslator, OllamaTranslator, XinferenceTranslator, AzureOpenAITranslator,
                           OpenAITranslator, ZhipuTranslator, ModelScopeTranslator, SiliconTranslator, GeminiTranslator, AzureTranslator, TencentTranslator,
                           DifyTranslator, AnythingLLMTranslator, ArgosTranslator, GrokTranslator, GroqTranslator, DeepseekTranslator, OpenAIlikedTranslator,
                           QwenMtTranslator, OpenRouterTranslator]:
            if service_name == translator.name:
                self.translator = translator(lang_in, lang_out, service_model, envs=envs, prompt=prompt, ignore_cache=ignore_cache)
        if not self.translator:
            raise ValueError("Unsupported translation service")

    def receive_layout(self, ltpage: LTPage):
        # Paragraph processing
        sstk: list[str] = []            # Paragraph text stack
        pstk: list[Paragraph] = []      # Paragraph properties stack
        vbkt: int = 0                   # Formula bracket count
        # Formula groups
        vstk: list[LTChar] = []         # Formula symbols group
        vlstk: list[LTLine] = []        # Formula lines group
        vfix: float = 0                 # Formula vertical offset
        # Formula group stacks
        var: list[list[LTChar]] = []    # Formula symbols stack
        varl: list[list[LTLine]] = []   # Formula lines stack
        varf: list[float] = []          # Formula vertical offsets stack
        vlen: list[float] = []          # Formula widths stack
        # Global
        lstk: list[LTLine] = []         # Global lines stack
        xt: LTChar = None               # Previous character
        xt_cls: int = -1                # Previous character's paragraph class, ensures new paragraph triggers regardless of first character's class
        vmax: float = ltpage.width / 4  # Maximum inline formula width
        ops: str = ""                   # Rendering result
        # Determine if the target language is RTL (e.g., Arabic)
        RTL_LANGS = {"ar", "he", "fa", "ur"}  # Right-to-left languages
        is_rtl = self.translator.lang_out.lower() in RTL_LANGS # Add other RTL languages as needed

        def vflag(font: str, char: str):    # Match formula (and subscript) fonts
            if isinstance(font, bytes):     # May not decode, convert directly to str
                try:
                    font = font.decode('utf-8')  # Attempt UTF-8 decoding
                except UnicodeDecodeError:
                    font = ""
            font = font.split("+")[-1]      # Truncate font name
            if re.match(r"\(cid:", char):
                return True
            # Font name based rules
            if self.vfont:
                if re.match(self.vfont, font):
                    return True
            else:
                if re.match(                                            # LaTeX fonts
                    r"(CM[^R]|MS.M|XY|MT|BL|RM|EU|LA|RS|LINE|LCIRCLE|TeX-|rsfs|txsy|wasy|stmary|.*Mono|.*Code|.*Ital|.*Sym|.*Math)",
                    font,
                ):
                    return True
            # Character set based rules
            if self.vchar:
                if re.match(self.vchar, char):
                    return True
            else:
                if (
                    char
                    and char != " "                                     # Not space
                    and (
                        unicodedata.category(char[0])
                        in ["Lm", "Mn", "Sk", "Sm", "Zl", "Zp", "Zs"]   # Letter modifiers, math symbols, separators
                        or ord(char[0]) in range(0x370, 0x400)          # Greek letters
                    )
                ):
                    return True
            return False

        ############################################################
        # A. Original Document Parsing
        for child in ltpage:
            if isinstance(child, LTChar):
                cur_v = False
                layout = self.layout[ltpage.pageid]
                # ltpage.height might be the height within fig, here we use layout.shape uniformly
                h, w = layout.shape
                # Read the current character's class in layout
                cx, cy = np.clip(int(child.x0), 0, w - 1), np.clip(int(child.y0), 0, h - 1)
                cls = layout[cy, cx]
                # Anchor bullet position in document
                if child.get_text() == "•":
                    cls = 0
                # Determine if current character belongs to formula
                if (                                                                                        # Determine if current character belongs to formula
                    cls == 0                                                                                # 1. Class is reserved area
                    or (cls == xt_cls and len(sstk[-1].strip()) > 1 and child.size < pstk[-1].size * 0.79)  # 2. Subscript font, with 0.76 for subscript and 0.799 for uppercase, using 0.79 as middle ground, considering initial letter magnification
                    or vflag(child.fontname, child.get_text())                                              # 3. Formula font
                    or (child.matrix[0] == 0 and child.matrix[3] == 0)                                      # 4. Vertical font
                ):
                    cur_v = True
                # Determine if bracket group belongs to formula
                if not cur_v:
                    if vstk and child.get_text() == "(":
                        cur_v = True
                        vbkt += 1
                    if vbkt and child.get_text() == ")":
                        cur_v = True
                        vbkt -= 1
                if (                                                        # Determine if current formula ends
                    not cur_v                                               # 1. Current character doesn't belong to formula
                    or cls != xt_cls                                        # 2. Current character and previous character don't belong to same paragraph
                    # or (abs(child.x0 - xt.x0) > vmax and cls != 0)        # 3. Line break within paragraph, could be a long italic paragraph or fraction line break, set threshold to distinguish
                    # Prevent pure formula (code) paragraph line breaks until text starts, ensuring only two cases:
                    # A. Pure formula (code) paragraph (absolute positioning) sstk[-1]=="" -> sstk[-1]=="{v*}"
                    # B. Text starting paragraph (relative positioning) sstk[-1]!=""
                    or (sstk[-1] != "" and abs(child.x0 - xt.x0) > vmax)    # Because cls==xt_cls==0 must have sstk[-1]=="", no need to check cls!=0 here
                ):
                    if vstk:
                        if (                                                # Adjust formula vertical offset based on text to the right
                            not cur_v                                       # 1. Current character doesn't belong to formula
                            and cls == xt_cls                               # 2. Current character and previous character belong to same paragraph
                            and child.x0 > max([vch.x0 for vch in vstk])    # 3. Current character is to the right of formula
                        ):
                            vfix = vstk[0].y0 - child.y0
                        if sstk[-1] == "":
                            xt_cls = -1 # Prevent pure formula paragraph (sstk[-1]=="{v*}") from continuing, but need to consider new character and subsequent connections, so modifying previous character's class

                        sstk[-1] += f"{{v{len(var)}}}"
                        var.append(vstk)
                        varl.append(vlstk)
                        varf.append(vfix)
                        vstk = []
                        vlstk = []
                        vfix = 0
                # Current character doesn't belong to formula or is first character of formula
                if not vstk:
                    if cls == xt_cls:               # Current character and previous character belong to same paragraph
                        if child.x0 > xt.x1 + 1:    # Add inline space
                            sstk[-1] += " "
                        elif child.x1 < xt.x0:      # Add line break space and mark original paragraph has line break
                            sstk[-1] += " "
                            pstk[-1].brk = True
                    else:                           # Create new paragraph based on current character
                        sstk.append("")
                        pstk.append(Paragraph(child.y0, child.x0, child.x0, child.x0, child.y0, child.y1, child.size, False))
                if not cur_v:                                               # Push text to stack
                    if (                                                    # Adjust paragraph properties based on current character
                        child.size > pstk[-1].size                          # 1. Current character larger than paragraph font
                        or len(sstk[-1].strip()) == 1                       # 2. Current character is second character of paragraph (considering initial letter magnification)
                    ) and child.get_text() != " ":                          # 3. Current character is not a space
                        pstk[-1].y -= child.size - pstk[-1].size            # Adjust paragraph initial vertical coordinate, assuming top alignment of characters with different sizes
                        pstk[-1].size = child.size
                    sstk[-1] += child.get_text()
                else:                                                       # Push formula to stack
                    if (                                                    # Adjust formula vertical offset based on text to the left
                        not vstk                                            # 1. Current character is first character of formula
                        and cls == xt_cls                                   # 2. Current character and previous character belong to same paragraph
                        and child.x0 > xt.x0                                # 3. Previous character is to the left of formula
                    ):
                        vfix = child.y0 - xt.y0
                    vstk.append(child)
                # Update paragraph boundaries, since line breaks within paragraphs may be followed by formula starts, handle outside
                pstk[-1].x0 = min(pstk[-1].x0, child.x0)
                pstk[-1].x1 = max(pstk[-1].x1, child.x1)
                pstk[-1].y0 = min(pstk[-1].y0, child.y0)
                pstk[-1].y1 = max(pstk[-1].y1, child.y1)
                # Update previous character
                xt = child
                xt_cls = cls
            elif isinstance(child, LTFigure):   # Figure
                pass
            elif isinstance(child, LTLine):     # Line
                layout = self.layout[ltpage.pageid]
                # ltpage.height might be the height within fig, here we use layout.shape uniformly
                h, w = layout.shape
                # Read the current line's class in layout
                cx, cy = np.clip(int(child.x0), 0, w - 1), np.clip(int(child.y0), 0, h - 1)
                cls = layout[cy, cx]
                if vstk and cls == xt_cls:      # Formula lines
                    vlstk.append(child)
                else:                           # Global lines
                    lstk.append(child)
            else:
                pass
        # Process ending
        if vstk:    # Push formula to stack
            sstk[-1] += f"{{v{len(var)}}}"
            var.append(vstk)
            varl.append(vlstk)
            varf.append(vfix)
        log.debug("\n==========[VSTACK]==========\n")
        for id, v in enumerate(var):  # Calculate formula width
            l = max([vch.x1 for vch in v]) - v[0].x0
            log.debug(f'< {l:.1f} {v[0].x0:.1f} {v[0].y0:.1f} {v[0].cid} {v[0].fontname} {len(varl[id])} > v{id} = {"".join([ch.get_text() for ch in v])}')
            vlen.append(l)

        ############################################################
        # B. Paragraph Translation
        log.debug("\n==========[SSTACK]==========\n")

        @retry(wait=wait_fixed(1))
        def worker(s: str):  # Multi-threaded translation
            if not s.strip() or re.match(r"^\{v\d+\}$", s):  # Don't translate empty strings and formulas
                return s
            try:
                new = self.translator.translate(s)
                return new
            except BaseException as e:
                if log.isEnabledFor(logging.DEBUG):
                    log.exception(e)
                else:
                    log.exception(e, exc_info=False)
                raise e

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.thread) as executor:
            news = list(executor.map(worker, sstk))

        # Post-process translated text for RTL languages
        if is_rtl:
            for i in range(len(news)):
                if not news[i].strip() or re.match(r"^\{v\d+\}$", news[i]):
                    continue  # Skip formulas and empty strings
                reshaped_text = arabic_reshaper.reshape(news[i])
                news[i] = bidi.algorithm.get_display(reshaped_text)
        ############################################################
        # Part C: New Document Layout
        def raw_string(fcur: str, cstk: str):  # Generate raw string for text rendering
            if fcur == self.noto_name:
                return "".join(["%04x" % self.noto.has_glyph(ord(c)) for c in cstk])
            elif isinstance(self.fontmap[fcur], PDFCIDFont):
                return "".join(["%04x" % ord(c) for c in cstk])
            else:
                return "".join(["%02x" % ord(c) for c in cstk])

        def gen_op_txt(font, size, x, y, rtxt):
            """
            Generates a PDF text operator string for text rendering.

            Example:
                >>> gen_op_txt("F1", 12, 100, 200, "48656C6C6F")
                "/F1 12.000000 Tf 1 0 0 1 100.000000 200.000000 Tm [<48656C6C6F>] TJ "
            """
            return f"/{font} {size:f} Tf 1 0 0 1 {x:f} {y:f} Tm [<{rtxt}>] TJ "

        def gen_op_line(x, y, xlen, ylen, linewidth):
            """
            Generates a PDF operator line for drawing.

            This function creates a string containing PDF operators for drawing a line
            between two points in a PDF document.
            """
            return f"ET q 1 0 0 1 {x:f} {y:f} cm [] 0 d 0 J {linewidth:f} w 0 0 m {xlen:f} {ylen:f} l S Q BT "

        # Define constants
        LANG_LINEHEIGHT_MAP = {
            "zh-cn": 1.4, "zh-tw": 1.4, "zh-hans": 1.4, "zh-hant": 1.4, "zh": 1.4,
            "ja": 1.1, "ko": 1.2, "en": 1.2, "ar": 2.3, "ru": 0.8, "uk": 0.8, "ta": 0.8
        }
        default_line_height = LANG_LINEHEIGHT_MAP.get(self.translator.lang_out.lower(), 1.1)
        ops_list = []

        # Initialize positioning helper
        pos_helper = PositionHelper(is_rtl, ltpage.width)

        for id, new in enumerate(news):
            # Get margins and indentation from PositionHelper
            actual_margin = pos_helper.get_margin()
            actual_indentation = pos_helper.get_indentation()
            # Enforce page margins
            x0 = pstk[id].x0#max(pstk[id].x0, actual_margin)                # Paragraph left boundary  
            x1 = pstk[id].x1#min(pstk[id].x1, ltpage.width - actual_margin) # Paragraph right boundary
            x = pos_helper.get_start_x(pstk[id].x, x1)          # Start at right for RTL, left for LTR  
            y = pstk[id].y                                      # Initial y coordinate
            height = pstk[id].y1 - pstk[id].y0                  # Paragraph height
            size = pstk[id].size - 1 if is_rtl else pstk[id].size  # Font size
            brk = pstk[id].brk                                  # Line break marker
            cstk = ""                                           # Current text stack
            fcur = self.noto_name if is_rtl else None           # Current font ID, set to default font
            lidx = 0                                            # Line index
            tx = x                                              # Text starting position
            fcur_ = fcur                                        # Initialize fcur_ to default font
            ptr = 0
            ops_vals = []

            log.debug(f"< {y} {x} {x0} {x1} {size} {brk} > {sstk[id]} | {new}")

            ch = None
            padding = None
            padding_len = 0
            context = re.split(r'( )', new) if is_rtl else new
            column_type = not pos_helper.has_formulas(new) if self.column_type == "one" else brk

            is_centered = pos_helper.is_centered_text(x0, x1, ltpage.width, brk)
            # if not brk:
            #     size = 12 if size < 12 else 0 # FIXME: Not sure if this is correct
            # Check for RTL text that needs padding, excluding centered text and formulas
            if is_rtl and lidx == 0 and not is_centered and column_type:

                # Calculate word widths accurately
                word_widths = [pos_helper.word_lengths(w, char_lengths_fn=self.noto.char_lengths, fontsize=size) 
                               for w in context]  # Only count non-empty words

                # Calculate available width per line
                if brk == False:
                    line_width = (x1 - x0) - (x1 - ltpage.width) # for section title
                else:
                    line_width = (x1 - x0)

                # Get total text width
                total_width = sum(word_widths)
                # Calculate number of full lines
                num_full_lines = math.floor(total_width / line_width)
                # Calculate remaining width more precisely
                remaining_width = total_width - (num_full_lines * line_width)
                # Calculate padding considering full lines
                padding_width = line_width - remaining_width
                if padding_width < line_width * 0.1:  # If padding too small
                    padding_width += line_width  # Add another line width

                if brk:
                    space_width = self.noto.char_lengths(' ', fontsize=size)[0]
                    num_spaces = math.ceil(padding_width / space_width) - (2 * actual_margin)
                    num_spaces -= 2 * actual_indentation
                else:
                    space_width = size / 3
                    num_spaces = math.ceil(padding_width / space_width)

                # Create padding and add to context
                padding = [' '] * num_spaces
                padding_len = len(padding)                
                context = padding + context

                # import pdb; pdb.set_trace()

            while ptr < len(context):
                vy_regex = re.match(r"\{\s*v([\d\s]+)\}", new[ptr:], re.IGNORECASE)
                mod = 0
                if vy_regex:  # Handle formula
                    ptr += len(vy_regex.group(0))
                    try:
                        vid = int(vy_regex.group(1).replace(" ", ""))
                        adv = vlen[vid]
                    except Exception:
                        continue
                    if var[vid][-1].get_text() and unicodedata.category(var[vid][-1].get_text()[0]) in ["Lm", "Mn", "Sk"]:
                        mod = var[vid][-1].width
                else:  # Handle text
                    ch = context[ptr]

                    fcur_ = None
                    try:
                        if fcur_ is None and self.fontmap["tiro"].to_unichr(ord(ch)) == ch:
                            fcur_ = "tiro"
                    except Exception:
                        pass
                    if fcur_ is None:
                        fcur_ = self.noto_name
                    # Calculate advance width
                    if fcur_ == self.noto_name:
                        if is_rtl:
                            adv = pos_helper.word_lengths(ch, char_lengths_fn=self.noto.char_lengths, fontsize=size)
                        else:
                            try:
                                adv = self.noto.char_lengths(ch, size)[0]
                            except Exception:
                                adv = 0
                    else:
                        adv = self.fontmap[fcur_].char_width(ord(ch)) * size
                    ptr += 1

                # Check for line break conditions
                at_line_end = pos_helper.is_at_line_end(x, adv, x0, x1 + 0.1 * size)
                if fcur_ != fcur or vy_regex or at_line_end:
                    if cstk:
                        ops_vals.append({
                            "type": OpType.TEXT,
                            "font": fcur,
                            "size": size,
                            "x": tx,
                            "dy": 0,
                            "rtxt": raw_string(fcur, cstk),
                            "lidx": lidx
                        })
                        cstk = ""

                if brk and pos_helper.needs_line_break(x, adv, x0, x1 + 0.1 * size):
                    x = pos_helper.get_line_start_x(x0, x1)
                    lidx += 1

                if vy_regex:  # Insert formula
                    fix = varf[vid] if fcur is not None else 0
                    for vch in var[vid]:
                        vc = chr(vch.cid)
                        ops_vals.append({
                            "type": OpType.TEXT,
                            "font": self.fontid[vch.font],
                            "size": vch.size,
                            "x": x + pos_helper.adjust_formula_offset(vch.x0 - var[vid][0].x0),
                            "dy": fix + vch.y0 - var[vid][0].y0,
                            "rtxt": raw_string(self.fontid[vch.font], vc),
                            "lidx": lidx
                        })
                    for l in varl[vid]:
                        if l.linewidth < 5:
                            ops_vals.append({
                                "type": OpType.LINE,
                                "x": l.pts[0][0] + x - var[vid][0].x0,
                                "dy": l.pts[0][1] + fix - var[vid][0].y0,
                                "linewidth": l.linewidth,
                                "xlen": l.pts[1][0] - l.pts[0][0],
                                "ylen": l.pts[1][1] - l.pts[0][1],
                                "lidx": lidx
                            })
                else:  # Insert text
                    if not cstk:
                        tx = x
                        if (x == x0 and ch == " " and not is_rtl) or (x == x1 and ch == " " and is_rtl):
                            adv = 0
                            if ch == " " and padding_len > 0:
                                cstk += " "
                                padding_len -= 1
                        else:
                            cstk += ch
                    else:
                        cstk += ch
                        if ch == " " and padding_len > 0:
                            cstk += " "
                            padding_len -= 1

                # Update position
                adv -= mod
                fcur = fcur_
                x = pos_helper.adjust_x_position(x, adv)  # Move left for RTL, right for LTR

            # Handle remaining text
            if cstk:
                ops_vals.append({
                    "type": OpType.TEXT,
                    "font": fcur,
                    "size": size,
                    "x": tx,
                    "dy": 0,
                    "rtxt": raw_string(fcur, cstk),
                    "lidx": lidx
                })

            # Mirror positions for RTL
            if is_rtl:
                for vals in ops_vals:
                    if vals["type"] == OpType.TEXT:
                        # Mirror x-position within paragraph bounds
                        vals["x"] = pos_helper.mirror_position(vals["x"], x0, x1)
                        if vals["lidx"] == 0:  # Indent first line leftward
                            vals["x"] -= actual_indentation
                    elif vals["type"] == OpType.LINE:
                        # Mirror line starting point
                        vals["x"] = pos_helper.mirror_position(vals["x"] + vals["xlen"], x0, x1) - vals["xlen"]
                
            # Adjust line height
            line_height = default_line_height
            total_lines = max((vals["lidx"] for vals in ops_vals), default=-1) + 1
            line_height_ = 1.35 if is_rtl else 1.0
            while (total_lines) * size * line_height > height and line_height >= line_height_:
                line_height -= 0.05

            # Generate operations
            for vals in ops_vals:
                adjusted_lidx = pos_helper.get_line_index(vals["lidx"], total_lines)
                y_pos = pos_helper.get_line_y_position(y, adjusted_lidx, size, line_height) + vals["dy"]
                if vals["type"] == OpType.TEXT:
                    ops_list.append(gen_op_txt(vals["font"], vals["size"], vals["x"], y_pos, vals["rtxt"]))
                elif vals["type"] == OpType.LINE:
                    ops_list.append(gen_op_line(vals["x"], y_pos, vals["xlen"], vals["ylen"], vals["linewidth"]))

        # Global lines with RTL adjustment
        for l in lstk:
            if l.linewidth < 5:
                x = l.pts[0][0]
                y = l.pts[0][1]
                xlen = l.pts[1][0] - l.pts[0][0]
                ylen = l.pts[1][1] - l.pts[0][1]
                if is_rtl:
                    x = ltpage.width - (x + xlen)
                ops_list.append(gen_op_line(x, y, xlen, ylen, l.linewidth))
        
        ops = f"BT {''.join(ops_list)}ET "
        return ops


class OpType(Enum):
    TEXT = "text"
    LINE = "line"


class PositionHelper:
    """Helper class to handle RTL/LTR positioning logic"""
    
    def __init__(self, is_rtl: bool, page_width: float):
        self.is_rtl = is_rtl
        self.page_width = page_width
        # Define margin constants
        self.MARGIN = 70
        self.RTL_MARGIN = 25
        self.INDENTATION = 20 
        self.RTL_INDENTATION = 10
        
    def get_margin(self):
        """Get appropriate margin based on text direction"""
        return self.RTL_MARGIN if self.is_rtl else self.MARGIN
        
    def get_indentation(self):
        """Get appropriate indentation based on text direction"""
        return self.RTL_INDENTATION if self.is_rtl else self.INDENTATION
        
    def get_start_x(self, x: float, x1: float):
        """Get starting x position based on text direction"""
        return x1 if self.is_rtl else x

    def adjust_x_position(self, x: float, advance: float):
        """Adjust x position based on text direction and advance width"""
        return x - advance if self.is_rtl else x + advance
        
    def mirror_position(self, x: float, x0: float, x1: float):
        """Mirror position within bounds for RTL text"""
        if self.is_rtl:
            return x0 + x1 - x
        return x
        
    def mirror_line_position(self, x: float, xlen: float):
        """Mirror line starting position for RTL"""
        if self.is_rtl:
            return self.page_width - (x + xlen)
        return x

    def get_line_start_x(self, x0: float, x1: float):
        """Get starting x position for a new line based on text direction"""
        return x1 if self.is_rtl else x0

    def is_at_line_end(self, x: float, advance: float, x0: float, x1: float) -> bool:
        """Check if current position + advance would go beyond line boundaries"""
        if self.is_rtl:
            return x - advance < x0
        return x + advance > x1

    def needs_line_break(self, x: float, advance: float, x0: float, x1: float) -> bool:
        """Check if a line break is needed based on position and advance width"""
        return self.is_at_line_end(x, advance, x0, x1)

    def adjust_formula_offset(self, offset: float) -> float:
        """Adjust formula x-offset based on text direction"""
        return -offset if self.is_rtl else offset
        
    def get_line_index(self, current_line: int, total_lines: int) -> int:
        """Calculate the correct line index based on text direction"""
        return (total_lines - current_line - 1) if self.is_rtl else current_line
        
    def get_line_y_position(self, y: float, line_idx: int, size: float, line_height: float) -> float:
        """Calculate y position based on line index and text direction"""
        return y - (line_idx * size * line_height)

    def word_lengths(self,
                     text,
                     char_lengths_fn,
                     fontsize=11,
                     language=None,
                     script=0,
                     wmode=0,
                     small_caps=0):
        """Calculate total width of a word with RTL text support.
        
        Parameters
        ----------
            text: The word to measure (unicode string)
            char_lengths_fn: Function to get individual character widths
            fontsize: Font size in points (default 11)
            language: Language code (e.g. 'ar' for Arabic)
            script: 0 for LTR, 1 for RTL (default 0)
            wmode: Writing mode (0=horizontal, 1=vertical)
            small_caps: Whether to use small caps (0=no, 1=yes)
            
        Returns
        -------
            float: Total width of the word including any RTL adjustments
        """
        # RTL languages we support
        RTL_LANGS = {'ar', 'he', 'fa', 'ur'}
        
        # Determine if text is RTL
        is_rtl = (script == 1 or 
                (language and language.lower() in RTL_LANGS) or
                any('\u0590' <= c <= '\u08FF' for c in text))  # Unicode RTL blocks
        
        # Get individual character widths
        char_widths = char_lengths_fn(text, fontsize, language, script, wmode, small_caps)
        total_width = sum(char_widths)
        
        # Add RTL spacing adjustment if needed
        if is_rtl:
            # Add 10% of font size as spacing adjustment for proper RTL rendering
            return total_width + (fontsize * 0.1)
        
        return total_width
    
                # Helper function to check if text is centered
    def is_centered_text(self, x0, x1, page_width, is_break):
        # Calculate margins from both sides
        left_margin = x0
        right_margin = page_width - x1
        # Allow small difference in margins (5% of page width)
        margin_tolerance = page_width * 0.05
        return abs(left_margin - right_margin) < margin_tolerance and not is_break

    # Helper function to check if text contains formulas
    def has_formulas(self, text):
        return bool(re.match(r"\{\s*v([\d\s]+)\}", text, re.IGNORECASE))
