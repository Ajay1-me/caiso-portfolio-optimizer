"""
CAISO OASIS public API client for Day-Ahead LMP data.

Endpoint: http://oasis.caiso.com/oasisapi/SingleZip
Query:    PRC_LMP (Locational Marginal Price), market_run_id=DAM

Response format:
  The API returns a ZIP archive containing a single XML file. Each
  <REPORT_DATA> element covers one hour/node/component combination.
  LMP is decomposed into three components per CAISO settlement rules
  (FERC Order 2000 / tariff Section 27):
    LMP  = total locational marginal price
    MCC  = marginal congestion component
    MCE  = marginal energy component (system lambda)
    MLC  = marginal loss component

  This client extracts only the LMP (total) rows.

Date convention:
  CAISO uses Hour Ending (HE) notation. The DAM clears the night before
  the operating day. We request startdatetime = operating_date T08:00-0000
  (midnight Pacific Standard, UTC−8) through T+1 08:00-0000 to capture
  all 24 operating hours.
"""

from __future__ import annotations

import io
import logging
import os
import zipfile
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

OASIS_URL = os.getenv("CAISO_OASIS_URL", "http://oasis.caiso.com/oasisapi/SingleZip")
TIMEOUT = int(os.getenv("CAISO_TIMEOUT_SEC", "30"))


def _build_query_params(op_date: date, node: str) -> dict:
    """
    Construct OASIS API query parameters for a given operating date.

    CAISO datetime format: YYYYMMDDThh:mm-0000 (UTC).
    Operating day = midnight-to-midnight Pacific → 08:00–08:00 UTC (PST offset).
    """
    start = datetime(op_date.year, op_date.month, op_date.day, 8, 0)
    end = start + timedelta(hours=24)
    fmt = "%Y%m%dT%H:%M-0000"
    return {
        "queryname": "PRC_LMP",
        "market_run_id": "DAM",
        "startdatetime": start.strftime(fmt),
        "enddatetime": end.strftime(fmt),
        "node": node,
        "version": "1",
    }


def _parse_lmp_xml(xml_bytes: bytes, node: str) -> pd.Series | None:
    """
    Parse CAISO OASIS PRC_LMP XML and return a 24-element Series (index=hour 0–23).

    The XML schema uses an <XML_DATA_ITEM> pattern where each <REPORT_DATA>
    block contains multiple key/value pairs. We filter for LMP_TYPE=LMP to
    get total price rather than individual components (MCC/MCE/MLC).
    """
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        log.warning("XML parse error: %s", exc)
        return None

    # Strip namespace prefix from all tags so we can match by local name.
    def _tag(el: ET.Element) -> str:
        return el.tag.split("}")[-1] if "}" in el.tag else el.tag

    prices: dict[int, float] = {}

    for report_data in root.iter():
        if _tag(report_data) != "REPORT_DATA":
            continue

        # Collect child element values into a flat dict.
        fields: dict[str, str] = {}
        data_items: dict[str, str] = {}

        for child in report_data:
            tag = _tag(child)
            if tag in ("OPR_DT", "OPR_HR", "NODE_ID", "NODE_ID_XML"):
                fields[tag] = (child.text or "").strip()
            elif tag == "XML_DATA_ITEM":
                item_name = item_val = ""
                for sub in child:
                    sub_tag = _tag(sub)
                    if sub_tag == "DATA_ITEM":
                        item_name = (sub.text or "").strip()
                    elif sub_tag == "XML_VALUE":
                        item_val = (sub.text or "").strip()
                if item_name:
                    data_items[item_name] = item_val

        node_id = fields.get("NODE_ID") or fields.get("NODE_ID_XML", "")
        lmp_type = data_items.get("LMP_TYPE", "")
        mw_str = data_items.get("MW", "")

        if node not in node_id or lmp_type != "LMP" or not mw_str:
            continue

        try:
            opr_hr = int(fields.get("OPR_HR", "0"))  # CAISO HE: 1–24
            prices[opr_hr - 1] = float(mw_str)       # convert to 0–23 index
        except ValueError:
            continue

    if not prices:
        log.warning("No LMP rows matched node=%s in XML response.", node)
        return None

    series = pd.Series(prices).reindex(range(24))
    if series.isna().any():
        log.warning("Missing hours in LMP response; interpolating.")
        series = series.interpolate(method="linear").ffill().bfill()

    return series


def fetch_lmp(op_date: str | date, node: str = "SP15_GEN-APND") -> np.ndarray | None:
    """
    Fetch 24-hour Day-Ahead LMP prices from CAISO OASIS for *op_date*.

    Returns a float64 ndarray of shape (24,) in $/MWh, or None on failure.
    Callers should fall back to sample data on None.

    Parameters
    ----------
    op_date:
        The operating date (date object or ISO string 'YYYY-MM-DD').
    node:
        CAISO pricing node identifier (default: SP15_GEN-APND).
    """
    if isinstance(op_date, str):
        op_date = date.fromisoformat(op_date)

    params = _build_query_params(op_date, node)
    log.info("Requesting OASIS LMP: node=%s date=%s", node, op_date)

    try:
        resp = requests.get(OASIS_URL, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as exc:
        log.warning("OASIS API request failed: %s", exc)
        return None

    # Response is a ZIP containing one XML file.
    try:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            xml_name = next(
                (n for n in zf.namelist() if n.endswith(".xml")), None
            )
            if xml_name is None:
                log.warning("No XML found inside OASIS ZIP response.")
                return None
            xml_bytes = zf.read(xml_name)
    except zipfile.BadZipFile:
        log.warning("OASIS response is not a valid ZIP (content-type: %s).", resp.headers.get("content-type"))
        return None

    series = _parse_lmp_xml(xml_bytes, node)
    return series.to_numpy(dtype=float) if series is not None else None
