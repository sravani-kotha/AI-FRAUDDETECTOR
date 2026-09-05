from pathlib import Path
from textwrap import wrap

OUTPUT = Path(__file__).parent.parent / "Fraud_Detector_5_Minute_Recording_Script.pdf"

sections = [
    ("Fraud Detector", "5-Minute Recording Script"),
    ("Before recording", "Open VS Code in the fraud-detector-starter folder. Open one PowerShell terminal in the project folder. Close unrelated tabs and notifications. Set browser zoom to 100% and place the browser beside VS Code. Start screen recording before running the first command."),
    ("0:00-0:35 | Introduce the project", "SCREEN: Keep VS Code on README.md. Move the cursor slowly over the project name, then over src/api.py in the Explorer.\n\nSAY: This is a defense-only checkout fraud risk engine. It takes transaction signals, estimates fraud risk, and returns one of three recommendations: approve, review, or decline. The project includes synthetic data generation, model training, evaluation, a FastAPI scoring service, and a browser dashboard for testing the API.\n\nCURSOR: Point to data, models, reports, src, and frontend as you mention them. Do not open lots of files."),
    ("0:35-1:15 | Explain the architecture", "SCREEN: Open ARCHITECTURE.md. Trace the diagram from Checkout transaction down to Reason codes + audit-friendly JSON response.\n\nSAY: The request first passes through feature validation. The same feature list is shared by training and serving, which prevents the model and API from silently using different columns. A gradient-boosted classifier produces a fraud probability. A threshold selected from validation data maps that probability to approve, review, or decline. Finally, the API returns the score, the decision, and short reason codes.\n\nCURSOR: Pause on the three decision labels and then on the reason-code output."),
    ("1:15-1:55 | Setup and tests", "SCREEN: Switch to PowerShell and run these commands one at a time:\n\npython -m src.generate_data --n 20000 --fraud-rate 0.02\npython -m src.train\npython -m src.evaluate\npython -m pytest -q\n\nSAY: The data in this demonstration is synthetic, so it is reproducible and safe for a walkthrough. The training step compares a baseline with a boosted model. Evaluation selects an operating threshold using business costs, and the test suite checks the core data and threshold behavior.\n\nThen run:\npython -m uvicorn src.api:app --reload --port 8000"),
    ("1:55-2:35 | Show the API and dashboard", "SCREEN: Open a browser at http://localhost:8000. Wait for the dashboard to load. Move the cursor over Model ready, the input panel, and the empty result panel.\n\nSAY: The backend serves the dashboard directly, so there is no separate frontend server. The green status indicator confirms that the trained model loaded at startup. On the left are the seven features accepted by the /score endpoint. On the right, the result panel will show the risk score, decision, and explanations.\n\nCURSOR: Point to Transaction input, Signal checks, and Model ready."),
    ("2:35-3:10 | Demonstrate approve", "SCREEN: Leave the default low-risk values: order value ratio 1.2, account age 180, transactions last hour 1, hour of day 14. Leave all signal checks off. Click Score transaction.\n\nSAY: This represents a normal established customer: a modest order, an older account, low transaction velocity, and no suspicious device or network signals. The model returns approve. The reason section confirms that there are no elevated risk signals.\n\nCURSOR: Move to the decision badge and risk percentage."),
    ("3:10-3:50 | Demonstrate decline", "SCREEN: Click Load high-risk example. Move over the changed values and the three selected signal checks. Click Score transaction.\n\nSAY: Now I am loading a deliberately high-risk example. The order is more than three times the customer average, the account is only four days old, there are five transactions in the last hour, and the session has a mismatch, VPN, and new-device signal. The model returns decline and lists the signals that drove the result.\n\nCURSOR: Pause over the risk percentage, decline badge, and each reason code."),
    ("3:50-4:25 | Demonstrate review", "SCREEN: Set order value ratio 2.1, account age 20, transactions last hour 3, hour of day 22. Turn on only New device. Leave mismatch and VPN off. Click Score transaction.\n\nSAY: Review is the middle path. It is useful when the score is not low enough for automatic approval but is not high enough for an automatic decline. This lets a merchant route an uncertain transaction to an analyst instead of treating every alert as fraud.\n\nCURSOR: Point to the review result and its shorter reason list."),
    ("4:25-5:00 | Metrics, limitations, and close", "SCREEN: Switch back to VS Code. Open reports/metrics.json, then briefly open src/features.py. Keep the cursor near the threshold and cost constants.\n\nSAY: The reported metrics come from an untouched test split, while the operating threshold is selected on validation data. The false-positive and false-negative costs are currently placeholders and must be replaced with a real merchant's numbers. The dataset is also synthetic, so these metrics demonstrate the pipeline but are not claims about live fraud performance. In production, the next steps would be real labeled data, stronger feature validation, monitoring, and a proper analyst review workflow. This completes the end-to-end fraud scoring demonstration."),
    ("Quick recovery notes", "If port 8000 is busy, run python -m uvicorn src.api:app --reload --port 8010 and open http://localhost:8010. If the model is missing, run python -m src.train before starting Uvicorn. If a command takes time, leave the terminal visible and explain what that step is doing. Never describe the synthetic metrics as production accuracy."),
]


def escape(text):
    return text.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')


def make_pdf():
    page_width, page_height = 612, 792
    margin = 48
    usable = page_width - 2 * margin
    pages = []
    lines = []
    for title, body in sections:
        lines.append((title, True))
        for paragraph in body.split('\n'):
            if not paragraph:
                lines.append(('', False))
                continue
            width = 78 if len(title) < 35 else 82
            for line in wrap(paragraph, width=width, break_long_words=False):
                lines.append((line, False))
        lines.append(('', False))

    current = []
    y = page_height - margin
    for text, heading in lines:
        height = 17 if heading else 12
        if y - height < margin:
            pages.append(current)
            current = []
            y = page_height - margin
        current.append((text, heading, y))
        y -= height
    if current:
        pages.append(current)

    objects = []
    def add(obj):
        objects.append(obj)
        return len(objects)

    font_regular = add('<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>')
    font_bold = add('<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>')
    page_ids = []
    content_ids = []
    for page in pages:
        commands = ['BT']
        for text, heading, y_pos in page:
            if heading:
                commands += [f'/F2 12 Tf {margin} {y_pos:.1f} Td', f'({escape(text)}) Tj', '0 -17 Td']
            elif text:
                commands += [f'/F1 9 Tf {margin} {y_pos:.1f} Td', f'({escape(text)}) Tj', '0 -12 Td']
            else:
                commands.append('0 -8 Td')
        commands.append('ET')
        stream = '\n'.join(commands)
        content_ids.append(add(f'<< /Length {len(stream.encode("latin-1"))} >>\nstream\n{stream}\nendstream'))

    pages_id = add('')
    for content_id in content_ids:
        page_ids.append(add(f'<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 {page_width} {page_height}] /Resources << /Font << /F1 {font_regular} 0 R /F2 {font_bold} 0 R >> >> /Contents {content_id} 0 R >>'))
    objects[pages_id - 1] = f'<< /Type /Pages /Kids [{" ".join(str(i) + " 0 R" for i in page_ids)}] /Count {len(page_ids)} >>'
    catalog_id = add(f'<< /Type /Catalog /Pages {pages_id} 0 R >>')

    output = bytearray(b'%PDF-1.4\n%\xe2\xe3\xcf\xd3\n')
    offsets = [0]
    for index, obj in enumerate(objects, 1):
        offsets.append(len(output))
        output += f'{index} 0 obj\n{obj}\nendobj\n'.encode('latin-1')
    xref = len(output)
    output += f'xref\n0 {len(objects) + 1}\n0000000000 65535 f \n'.encode('latin-1')
    for offset in offsets[1:]:
        output += f'{offset:010d} 00000 n \n'.encode('latin-1')
    output += f'trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\nstartxref\n{xref}\n%%EOF\n'.encode('latin-1')
    OUTPUT.write_bytes(output)
    print(f'Created {OUTPUT} ({len(pages)} pages)')


if __name__ == '__main__':
    make_pdf()
