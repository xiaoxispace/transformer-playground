import re

from lxml import etree

class _EEBOTCPPreprocessor:

    # ---------- helpers ----------

    @staticmethod
    def _text_of(elem, skip_tags=("NOTE", "PB", "GAP")):
        """Concatenate all text in elem, skipping certain tags."""
        if elem is None:
            return ""
        parts = []
        if elem.text:
            parts.append(elem.text)
        for child in elem:
            tag = etree.QName(child).localname
            if tag in skip_tags:
                if child.tail:
                    parts.append(child.tail)
                continue
            parts.append(
                _EEBOTCPPreprocessor._text_of(child, skip_tags)
            )
            if child.tail:
                parts.append(child.tail)
        return "".join(parts)

    @staticmethod
    def _clean(s):
        return " ".join(s.split())

    @classmethod
    def _format_stage(cls, stage_elem):
        txt = cls._clean(cls._text_of(stage_elem))
        if not txt:
            return ""
        if not txt.endswith((".", "!", "?")):
            txt += "."
        return f"[{txt}]"

    # ---------- dramatis personae ----------

    @classmethod
    def _format_dramatis_personae(cls, dp_list):
        lines = ["Characters in the Play"]
        for item in dp_list.findall("ITEM"):
            raw = cls._clean(cls._text_of(item)).removesuffix(".")
            if raw:
                lines.append(raw)
        return "\n".join(lines)

    # ---------- acts / scenes ----------

    @classmethod
    def _parse_scene(cls, scene_div):
        blocks = []

        head = scene_div.find("HEAD")
        if head is not None:
            blocks.append(cls._clean(cls._text_of(head)).rstrip(".:"))

        for child in scene_div:
            tag = etree.QName(child).localname
            if tag == "HEAD":
                continue
            if tag == "STAGE":
                sd = cls._format_stage(child)
                if sd:
                    blocks.append(sd)
                continue
            if tag == "SP":
                speaker = child.find("SPEAKER")
                speaker_name = (
                    cls._clean(cls._text_of(speaker)) if speaker is not None else ""
                ).removesuffix(".")

                spoken_parts = []
                for sub in child:
                    sub_tag = etree.QName(sub).localname
                    if sub_tag == "SPEAKER":
                        continue
                    if sub_tag == "L":
                        line_txt = cls._clean(cls._text_of(sub))
                        if line_txt:
                            spoken_parts.append(line_txt)
                        for st in sub.findall("STAGE"):
                            sd = cls._format_stage(st)
                            if sd:
                                spoken_parts.append(sd)
                    elif sub_tag == "P":
                        p_txt = cls._clean(cls._text_of(sub))
                        if p_txt:
                            spoken_parts.append(p_txt)
                        for st in sub.findall("STAGE"):
                            sd = cls._format_stage(st)
                            if sd:
                                spoken_parts.append(sd)
                    elif sub_tag == "STAGE":
                        sd = cls._format_stage(sub)
                        if sd:
                            spoken_parts.append(sd)
                if spoken_parts:
                    spoken_parts[-1] = spoken_parts[-1] + "\n"
                if speaker_name:
                    blocks.append(speaker_name)
                blocks.extend(spoken_parts)
        return blocks

    @classmethod
    def _parse_act(cls, act_div):
        blocks = []
        head = act_div.find("HEAD")
        if head is not None:
            blocks.append(cls._clean(cls._text_of(head)).rstrip(".:").upper())
        for scene in act_div.findall("DIV2[@TYPE='scene']"):
            blocks.append("")
            blocks.extend(cls._parse_scene(scene))
        return blocks

    # ---------- top-level ----------

    @classmethod
    def _preprocess_regularized_EEBO_TCP(cls, file_path):
        tree = etree.parse(str(file_path))
        root = tree.getroot()

        out_blocks = []

        dp_div = root.find('.//DIV1[@TYPE="dramatis personae"]')
        if dp_div is not None:
            dp_list = dp_div.find("LIST")
            if dp_list is not None:
                out_blocks.append(cls._format_dramatis_personae(dp_list))
        out_blocks.append("")

        body = root.find(".//BODY")
        if body is None:
            raise ValueError(f"No BODY found in {file_path}")

        for act in body.findall("DIV1[@TYPE='act']"):
            out_blocks.append("")
            out_blocks.extend(cls._parse_act(act))

        return "\n".join(out_blocks).strip() + "\n"


class _SHCPreprocessor:

    # ---------- helpers ----------

    @staticmethod
    def _w_text(w):
        reg = w.get("reg")
        return reg if reg is not None else (w.text or "")

    @classmethod
    def _text_of(cls, elem, skip_tags=("note", "pb")):
        """Tag-aware flatten; prefers `reg` on <w> tokens.

        Returns raw text with tokens separated; punctuation spacing is
        normalized afterwards by _clean.
        """
        if elem is None:
            return ""
        tag = etree.QName(elem).localname
        if tag in skip_tags:
            return elem.tail or ""
        if tag == "w":
            return cls._w_text(elem) + (elem.tail or "")
        if tag == "pc":
            # punctuation token: attach to preceding word (no leading space)
            return (elem.text or "") + (elem.tail or "")
        if tag == "c":
            return " " + (elem.tail or "")

        parts = []
        if elem.text:
            parts.append(elem.text)
        for child in elem:
            ctag = etree.QName(child).localname
            if ctag in skip_tags:
                if child.tail:
                    parts.append(child.tail)
                continue
            parts.append(cls._text_of(child, skip_tags))
            if child.tail:
                parts.append(child.tail)
        return "".join(parts)

    _NO_SPACE_BEFORE = set(".,;:!?)]}…’”\"'")
    _NO_SPACE_AFTER = set("([{“‘\"")

    @classmethod
    def _clean(cls, s):
        """Normalize whitespace and fix punctuation spacing."""
        # 1) collapse all whitespace runs to single spaces
        s = " ".join(s.split())
        # 2) remove space before closing punctuation
        s = re.sub(r"\s+([.,;:!?)\]}\u2026\u2019\u201d\"'])", r"\1", s)
        # 3) remove space after opening punctuation
        s = re.sub(r"([(\[{\u2018\u201c\"'])\s+", r"\1", s)
        return s

    @classmethod
    def _format_stage(cls, stage_elem):
        txt = cls._clean(cls._text_of(stage_elem))
        if not txt:
            return ""
        if not txt.endswith((".", "!", "?")):
            txt += "."
        return f"[{txt}]"

    @staticmethod
    def _strip_namespaces(root):
        for elem in root.iter():
            if isinstance(elem.tag, str):
                elem.tag = etree.QName(elem).localname
        etree.cleanup_namespaces(root)
        return root

    # ---------- dramatis personae ----------

    @classmethod
    def _format_dramatis_personae(cls, cast_div):
        """SHC cast list lives in <back>; same output shape as EEBO TCP."""
        lines = ["Characters in the Play"]
        for item in cast_div.findall(".//item"):
            name_seg = item.find("seg[@type='character_name']")
            if name_seg is None:
                continue
            raw = cls._clean(cls._text_of(name_seg))
            if "-" in raw:
                raw = raw.split("-", 1)[1]
            raw = raw.replace("_", " ").strip()
            if raw:
                lines.append(raw.upper())
        return "\n".join(lines)

    # ---------- acts / scenes ----------

    @classmethod
    def _parse_scene(cls, scene_div):
        """SHC has no act/scene splits; a scene_div is a top-level <div>."""
        blocks = []

        head = scene_div.find("head")
        if head is not None:
            head_txt = cls._clean(cls._text_of(head))
            if head_txt:
                blocks.append(head_txt.rstrip(".:"))

        for child in scene_div:
            tag = etree.QName(child).localname
            if tag == "head":
                continue
            if tag == "stage":
                sd = cls._format_stage(child)
                if sd:
                    blocks.append(sd)
                continue
            if tag == "sp":
                blocks.extend(cls._parse_sp(child))
                continue
            if tag == "l":
                line = cls._clean(cls._text_of(child))
                if line:
                    blocks.append(line)
                continue
            if tag == "p":
                para = cls._clean(cls._text_of(child))
                if para:
                    blocks.append(para)
                continue
            if tag == "trailer":
                tr = cls._clean(cls._text_of(child))
                if tr:
                    blocks.append(tr)
                continue
        return blocks

    @classmethod
    def _parse_sp(cls, sp):
        """Speaker + spoken lines. Mirrors the SP branch of EEBO _parse_scene."""
        blocks = []
        speaker = sp.find("speaker")
        speaker_name = (
            cls._clean(cls._text_of(speaker)) if speaker is not None else ""
        ).removesuffix(".")

        spoken = []
        for sub in sp:
            stag = etree.QName(sub).localname
            if stag == "speaker":
                continue
            if stag in ("l", "p"):
                txt = cls._clean(cls._text_of(sub))
                if txt:
                    spoken.append(txt)
                for st in sub.findall("stage"):
                    sd = cls._format_stage(st)
                    if sd:
                        spoken.append(sd)
            elif stag == "stage":
                sd = cls._format_stage(sub)
                if sd:
                    spoken.append(sd)
        if len(spoken):
            # each speaker are separated by a newline
            spoken[-1] = spoken[-1] + '\n'

        if speaker_name:
            blocks.append(speaker_name)
        blocks.extend(spoken)
        return blocks

    @classmethod
    def _parse_act(cls, body_div):
        """SHC has no <DIV1 TYPE='act'>; treat each body <div> as one block."""
        head = body_div.find("head")
        blocks = []
        if head is not None:
            head_txt = cls._clean(cls._text_of(head))
            if head_txt:
                blocks.append(head_txt.rstrip(".:").upper())
        blocks.extend(cls._parse_scene(body_div))
        return blocks

    # ---------- top-level ----------

    @classmethod
    def _preprocess_shc(cls, file_path):
        tree = etree.parse(str(file_path))
        root = cls._strip_namespaces(tree.getroot())

        out_blocks = []

        cast_div = root.find(".//div[@type='machine-generated_castlist']")
        if cast_div is not None:
            out_blocks.append(cls._format_dramatis_personae(cast_div))
        out_blocks.append("")

        body = root.find(".//body")
        if body is None:
            raise ValueError(f"No BODY found in {file_path}")

        for body_div in body.findall("div"):
            out_blocks.append("")
            out_blocks.extend(cls._parse_act(body_div))

        return "\n".join(out_blocks).strip() + "\n"
