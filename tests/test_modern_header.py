"""Header parsing for the 2020s Indian Kanoon layout (from the 2025 samples)."""

import pytest

from kg_create import clean_page_text, parse_header_metadata, role_label


def pdf_pages(*raw_pages):
    return [
        {"page": number, "text": clean_page_text(raw), "raw_text": raw}
        for number, raw in enumerate(raw_pages, start=1)
    ]


CHOKSHI = """Ac Chokshi Share Broker Private Limited vs Jatin Pratap Desai
on 10 February, 2025
Author: Pamidighantam Sri Narasimha
Bench: Pamidighantam Sri Narasimha
2025 INSC 174                                                             REPORTABLE
                                     IN THE SUPREME COURT OF INDIA
                                      CIVIL APPELLATE JURISDICTION
                                     CIVIL APPEAL NO.       OF 2025
                                ARISING OUT OF SLP (C) No. 18393 OF 2021
             AC CHOKSHI SHARE BROKER
             PRIVATE LIMITED                                             ...APPELLANT(S)
                                                   VERSUS
               JATIN PRATAP DESAI & ANR.                               …RESPONDENT(S)
                                               JUDGMENT
PAMIDIGHANTAM SRI NARASIMHA, J.
1. Leave granted.
Ac Chokshi Share Broker Private Limited vs Jatin Pratap Desai on 10 February, 2025
Indian Kanoon - http://indiankanoon.org/doc/120709208/
1
"""

CHOKSHI_LAST = """35. No order as to costs.
……………………………....J.
[PAMIDIGHANTAM SRI NARASIMHA]
……………………………....J.
[SANDEEP MEHTA]
NEW DELHI;
FEBRUARY 10, 2025
"""

KENNEDY = """A John Kennedy vs The State Of Tamil Nadu on 24 March, 2025
Author: Vikram Nath
Bench: Vikram Nath
2025 INSC 443                                                             REPORTABLE
                                      IN THE SUPREME COURT OF INDIA
                                       CIVIL APPELLATE JURISDICTION
                                      SLP(CIVIL) NO(S). 999-1001 OF 2025
            A. JOHN KENNEDY ETC.                                          …PETITIONER(S)
                                                    VERSUS
            STATE OF TAMIL NADU
            AND OTHERS ETC.                                           ...RESPONDENT(S)
                                                  ORDER
Mehta, J.
1. These petitions raise two important issues for our consideration. The first being the
erstwhile tea estate by the name of Bombay Burma Trading
Corporation Limited [for short ‘BBTCL’] located in 10:38:01 IST Reason:
Singampatti, Tamil Nadu.
"""

KENNEDY_LAST = """34. List on 22nd April, 2025.
…………………………J.
(VIKRAM NATH)
…………………………J.
(SANDEEP MEHTA)
NEW DELHI;
"""

# Multi-matter header with a case-number prefix on the party line, the
# digital-signature stamp in the block and a letter-spaced signature.
BHANOT = """Rakesh Bhanot vs M/S Gurdas Agro Pvt. Ltd on 1 April, 2025
2025 INSC 445 REPORTABLE
IN THE SUPREME COURT OF INDIA
CRIMINAL APPELLATE/CIVIL ORIGINAL JURISDICTION
CRIMINAL APPEAL NO. 1607 OF 2025
(@ SPECIAL LEAVE PETITION (CRL.) NO. 6087 OF 2023) RAKESH BHANOT … APPELLANT(s)
-VERSUS-
M/S.GURDAS AGRO PVT. LTD. … RESPONDENT(S)
WITH
CRIMINAL APPEAL NO. 1608 OF 2025
Signature Not Verified
Digitally signed by
CHANDRESH
Date: 2025.04.04
16:28:38 IST
Reason:
"""

BHANOT_LAST = """21. Pending application(s), if any, shall stand closed.
.....................................J. [J.B. Pardiwala]
. . . . . . . . . J . ( V I K R A M N A T H )
NEW DELHI
"""


def test_modern_header_chokshi():
    header = parse_header_metadata(pdf_pages(CHOKSHI, CHOKSHI_LAST))
    assert header["name"] == "Ac Chokshi Share Broker Private Limited v. Jatin Pratap Desai"
    assert header["source_title"] == (
        "Ac Chokshi Share Broker Private Limited vs Jatin Pratap Desai on 10 February, 2025"
    )
    assert header["date"] == "2025-02-10"
    assert header["court"] == "Supreme Court of India"
    assert header["citations"] == ["2025 INSC 174"]
    # The Bench line lists one judge; the signature block has the full bench.
    assert header["judges"] == ["Pamidighantam Sri Narasimha", "Sandeep Mehta"]
    assert header["header_judges"] == ["Pamidighantam Sri Narasimha"]
    assert header["author"] == "Pamidighantam Sri Narasimha"
    assert [(p["role"], p["name"]) for p in header["parties"]] == [
        ("APPELLANT", "Ac Chokshi Share Broker Private Limited"),
        ("RESPONDENT", "Jatin Pratap Desai"),
    ]
    assert header["parties"][1]["evidence"] == "JATIN PRATAP DESAI & ANR. …RESPONDENT(S)"


def test_modern_header_body_author_overrides_indian_kanoon_author():
    header = parse_header_metadata(pdf_pages(KENNEDY, KENNEDY_LAST))
    assert header["header_author"] == "Vikram Nath"
    assert header["author"] == "Sandeep Mehta"
    assert header["judges"] == ["Vikram Nath", "Sandeep Mehta"]
    assert [(p["role"], p["name"]) for p in header["parties"]] == [
        ("PETITIONER", "A. John Kennedy"),
        ("RESPONDENT", "State of Tamil Nadu"),
    ]
    assert header["name"] == "A. John Kennedy v. State of Tamil Nadu"


def test_modern_header_multi_matter_with_signature_stamp():
    header = parse_header_metadata(pdf_pages(BHANOT, BHANOT_LAST))
    assert header["date"] == "2025-04-01"
    assert header["citations"] == ["2025 INSC 445"]
    assert header["author"] == ""
    assert header["judges"] == ["J.B. Pardiwala", "Vikramnath"]
    assert [(p["role"], p["name"]) for p in header["parties"]] == [
        ("APPELLANT", "Rakesh Bhanot"),
        ("RESPONDENT", "M/S.Gurdas Agro Pvt. Ltd"),
    ]


def test_letter_spaced_signature_resolved_against_bench_line():
    raw = BHANOT.replace("2025 INSC 445", "Bench: Vikram Nath\n2025 INSC 445")
    header = parse_header_metadata(pdf_pages(raw, BHANOT_LAST))
    assert header["judges"] == ["J.B. Pardiwala", "Vikram Nath"]


def test_modern_header_title_spanning_lines_and_no_role_labels():
    raw = """Cmj Foundation vs The State Of Meghalaya on 13 February,
2025
Author: Vikram Nath
Bench: Vikram Nath, Sanjay Karol
2025 INSC 211
SUPREME COURT OF INDIA
CIVIL APPELLATE JURISDICTION
CIVIL APPEAL NO(S). 9694 OF 2024
CMJ FOUNDATION AND OTHERS
VERSUS
STATE OF MEGHALAYA AND OTHERS
WITH
CIVIL APPEAL NO(S). 9695 OF 2024
JUDGMENT
"""
    header = parse_header_metadata(pdf_pages(raw))
    assert header["date"] == "2025-02-13"
    assert header["court"] == "Supreme Court of India"
    assert header["judges"] == ["Vikram Nath", "Sanjay Karol"]
    assert [(p["role"], p["name"]) for p in header["parties"]] == [
        ("APPELLANT", "Cmj Foundation"),
        ("RESPONDENT", "State of Meghalaya"),
    ]


def test_modern_header_without_party_block_uses_title_not_page_text():
    raw = "Jamin vs State Of Uttar Pradesh on 6 March, 2025\n2025 INSC 330\n" + "text " * 400
    header = parse_header_metadata(pdf_pages(raw))
    assert header["parties"] == []
    assert header["name"] == "Jamin v. State Of Uttar Pradesh"
    assert len(header["source_title"]) < 100


def test_high_court_heading():
    raw = """X vs Y on 1 May, 2024
IN THE HIGH COURT OF DELHI AT NEW DELHI
X ...PETITIONER
VERSUS
Y ...RESPONDENT
"""
    assert parse_header_metadata(pdf_pages(raw))["court"] == "High Court of Delhi"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("ANMOL APPELLANT(s)", ("ANMOL", "APPELLANT")),
        ("DALIP RAM …APPELLANT(S)", ("DALIP RAM", "APPELLANT")),
        ("X ....RESPO", ("X", "RESPONDENT")),
        ("DALJIT SINGH … A", ("DALJIT SINGH", "APPELLANT")),
        ("VIJAY PRABHU Peti", ("VIJAY PRABHU", "PETITIONER")),
        ("H. N. PANDAKUMAR ...Applicant(S)/", ("H. N. PANDAKUMAR", "PETITIONER")),
        (
            "CUDDALORE POWERGEN …APPELLANT(S) CORPORATION LTD",
            ("CUDDALORE POWERGEN CORPORATION LTD", "APPELLANT"),
        ),
        ("STATE OF KERELA Respondent(s)…….", ("STATE OF KERELA", "RESPONDENT")),
        ("K. A. RAJU", None),
        ("M/S. R. K. TRADERS", None),
        ("SPECIAL LEAVE PETITION (C) NO. 1", None),
    ],
)
def test_role_label(text, expected):
    assert role_label(text) == expected


def test_clean_page_text_strips_signature_stamp():
    raw = (
        "Corporation Limited located in 10:38:01 IST Reason:\n"
        "Singampatti.\n"
        "Signature Not Verified\n"
        "Digitally signed by\n"
        "NIDHI AHUJA\n"
        "Date: 2025.01.10\n"
        "16:28:38 IST\n"
        "Reason:\n"
        "2. The appeal\n"
    )
    assert clean_page_text(raw) == "Corporation Limited located in Singampatti. 2. The appeal"
