"""업로드한 정답 PDF와 OMR을 읽고, Python으로 채점합니다."""
from io import BytesIO
import base64
import json
import re
import unicodedata

import pandas as pd
import pdfplumber
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

ENGINE_VERSION = "2026-11-08-factual-grading-1"


def choice_number(value):
    text = unicodedata.normalize("NFKC", str(value)).strip()
    return int(float(text)) if re.fullmatch(r"[1-5](?:\.0)?", text) else None


def read_answer_key(pdf_bytes):
    """정답표의 반복된 '문항 번호 / 정답 / 배점' 열을 실제 PDF에서 읽습니다."""
    answers = {}
    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            # 먼저 표의 셀을 읽습니다. 선이 없는 정답표는 아래 텍스트 방식으로 읽습니다.
            for table in page.extract_tables():
                for row in table:
                    for start in range(0, len(row) - 2, 3):
                        q, answer, points = row[start:start + 3]
                        if not q or not str(q).strip().isdigit():
                            continue
                        answer = choice_number(answer)
                        if answer is not None and re.fullmatch(r"\d+(?:\.0)?", str(points).strip()):
                            item = {"answer": answer, "points": int(float(points))}
                            number = int(q)
                            if number in answers and answers[number] != item:
                                raise ValueError(f"정답표에서 {number}번의 값이 서로 다릅니다.")
                            answers[number] = item
            text = unicodedata.normalize("NFKC", page.extract_text() or "")
            for line in text.splitlines():
                values = line.split()
                if not values or len(values) % 3:
                    continue
                if not all(re.fullmatch(r"\d+", v) for v in values):
                    continue
                for start in range(0, len(values), 3):
                    q, answer, points = map(int, values[start:start + 3])
                    if q > 0 and 1 <= answer <= 5 and 0 < points <= 100:
                        item = {"answer": answer, "points": points}
                        if q in answers and answers[q] != item:
                            raise ValueError(f"정답표에서 {q}번의 값이 서로 다릅니다.")
                        answers[q] = item
    if not answers or sorted(answers) != list(range(1, max(answers) + 1)):
        raise ValueError("정답 PDF에서 연속된 문항 번호·정답·배점을 읽지 못했습니다.")
    if any(item["points"] <= 0 for item in answers.values()):
        raise ValueError("정답표의 배점은 양수여야 합니다.")
    return dict(sorted(answers.items()))


def read_omr(file_bytes, filename, question_numbers):
    source = BytesIO(file_bytes)
    if filename.lower().endswith(".csv"):
        try:
            df = pd.read_csv(source, dtype=object, encoding="utf-8-sig")
        except UnicodeDecodeError:
            source.seek(0)
            df = pd.read_csv(source, dtype=object, encoding="cp949")
    else:
        df = pd.read_excel(source, dtype=object)
    df = df.dropna(how="all")
    if df.empty:
        raise ValueError("OMR에 학생 데이터가 없습니다.")
    name_col = next((c for c in df.columns if str(c).strip() in
                     ["학생명", "학생", "이름", "성명", "학번", "학생번호", "student", "student_id"]), None)
    if name_col is None:
        raise ValueError("OMR에는 학생명·이름·학번·학생번호 중 하나의 열이 필요합니다.")
    names = df[name_col].astype("string").str.strip()
    if names.isna().any() or (names == "").any() or names.duplicated().any():
        raise ValueError("학생 이름/번호가 비어 있거나 중복됩니다. 고유한 학생 식별자를 사용하세요.")
    columns = {}
    for col in df.columns:
        if re.fullmatch(r"(?:[Qq]|문항)\s*0*\d+(?:\s*번)?\.\d+", str(col).strip()):
            raise ValueError(f"OMR 응답 열이 중복되었거나 문항 번호가 잘못되었습니다: {col}")
        match = re.fullmatch(r"(?:[Qq]\s*0*|문항\s*0*)?(\d+)(?:\s*번)?", str(col).strip())
        if match:
            number = int(match.group(1))
            if number in columns:
                raise ValueError(f"OMR에 {number}번 응답 열이 중복됩니다.")
            columns[number] = col
    if set(columns) != set(question_numbers):
        missing = sorted(set(question_numbers) - set(columns))
        extra = sorted(set(columns) - set(question_numbers))
        raise ValueError(f"정답표와 OMR 문항이 다릅니다. 누락: {missing}, 추가: {extra}")
    responses = pd.DataFrame(index=range(len(df)))
    for q in question_numbers:
        values = []
        for name, raw in zip(names, df[columns[q]]):
            blank = pd.isna(raw) or str(raw).strip() in ["", "0", "0.0", "-", "미응답"]
            number = None if blank else choice_number(raw)
            if not blank and number is None:
                raise ValueError(f"{name}의 {q}번 응답이 1~5 또는 미응답 값이 아닙니다.")
            values.append(number or 0)
        responses[q] = values
    return names.tolist(), responses


def read_exam(pdf_bytes):
    """2단 시험지를 열별로 읽고, 공통 지문도 해당 문항과 함께 보존합니다."""
    questions = {}
    shared = ""
    shared_start = shared_end = 0
    shared_started = False
    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            for left, right in [(0, page.width / 2), (page.width / 2, page.width)]:
                column = page.crop((left, 0, right, page.height))
                text = column.extract_text() or ""
                text = text.split("* 확인 사항")[0]
                text = re.sub(r"(?m)^.*이 문제지에 관한 저작권.*$", "", text)
                markers = list(re.finditer(r"(?m)^\s*(\d{1,3})\.\s|\[(\d{1,3})\s*[~～∼–-]\s*(\d{1,3})\]", text))
                # 공통 지문이 다음 열까지 이어지면 첫 문항 전의 나머지 본문도 붙입니다.
                if shared and not shared_started and markers and markers[0].group(1):
                    first = int(markers[0].group(1))
                    if shared_start <= first <= shared_end:
                        shared += "\n" + text[:markers[0].start()].strip()
                for i, match in enumerate(markers):
                    end = markers[i + 1].start() if i + 1 < len(markers) else len(text)
                    section = text[match.start():end].strip()
                    if match.group(2):
                        shared = section
                        shared_start = int(match.group(2))
                        shared_end = int(match.group(3))
                        shared_started = False
                        continue
                    number = int(match.group(1))
                    if number in questions:
                        raise ValueError(f"문제지에 {number}번 문항이 중복됩니다. 같은 페이지나 다른 시험이 섞이지 않았는지 확인해 주세요.")
                    if number > shared_end:
                        shared = ""
                    shared_started = True
                    questions[number] = {"page": page_num, "text": (shared + "\n" + section).strip()}
                    # 그림·도표 문항은 원본을 바꾸지 않고 메모리에서만 참조용 이미지를 읽습니다.
                    if re.search(r"그림|도표|표를", section[:130]):
                        words = column.extract_words()
                        starts = [w["top"] for w in words if w["text"] == f"{number}."]
                        next_number = next((int(m.group(1)) for m in markers[i+1:] if m.group(1)), None)
                        ends = [w["top"] for w in words if next_number and w["text"] == f"{next_number}."]
                        top = max(0, min(starts) - 3) if starts else 0
                        bottom = min(ends) - 3 if ends else page.height - 20
                        image = page.crop((left, top, right, max(top + 1, bottom))).to_image(resolution=200).original
                        buffer = BytesIO(); image.save(buffer, format="PNG")
                        questions[number]["image_url"] = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    if not questions:
        raise ValueError("문제 PDF에서 문항을 읽지 못했습니다. 텍스트를 선택할 수 있는 PDF를 사용하세요.")
    return questions


def file_warnings(exam_bytes, answer_bytes, exam_name="", answer_name=""):
    def identity(data, name):
        with pdfplumber.open(BytesIO(data)) as pdf:
            heading=(pdf.pages[0].extract_text() or '')[:1800]
        heading=re.sub(r'\s+','',heading)
        year=re.search(r'(20\d{2})학년도',heading)
        form=re.search(r'(홀수|짝수)\)?형',heading)
        subject_heading=heading[:300].replace('대학수학능력시험','')
        subject_name=name.replace('대학수학능력시험','')
        subject=next((s for s in ['영어','국어','수학','한국사'] if s in subject_heading or s in subject_name),None)
        exam_type=next((t for t in ['대학수학능력시험','전국연합학력평가','모의평가','학업성취도평가'] if t in heading[:400] or t in name),None)
        return {'학년도':year.group(1) if year else None,'시험형':form.group(1) if form else None,'과목':subject,'시험명':exam_type}
    exam,answer=identity(exam_bytes,exam_name),identity(answer_bytes,answer_name)
    warnings=[]
    for label in exam:
        if exam[label] and answer[label] and exam[label]!=answer[label]:
            warnings.append(f"문제지와 정답표의 {label} 표기가 다릅니다(문제지: {exam[label]}, 정답표: {answer[label]}). 동일 시험인지 확인하세요. 문항 수가 같아 계산은 진행했지만 결과가 잘못될 수 있습니다.")
    warnings.append("문항 번호 일치는 검사했지만, OMR에 시험명·학년도 식별 정보가 없어 동일 시험의 응답인지까지 확인할 수는 없습니다. 업로드한 원본 시험을 확인하세요.")
    return warnings


def analyze(exam_bytes, answer_bytes, omr_bytes, omr_name, exam_name="", answer_name=""):
    key = read_answer_key(answer_bytes)
    names, responses = read_omr(omr_bytes, omr_name, key)
    questions = read_exam(exam_bytes)
    if set(key) != set(questions):
        raise ValueError(f"채점을 중단했습니다. 문제지 {len(questions)}문항과 정답표 {len(key)}문항의 번호가 다릅니다. "
                         f"본문을 읽지 못한 번호: {sorted(set(key)-set(questions))}, 정답이 없는 번호: {sorted(set(questions)-set(key))}. "
                         "다른 시험·누락 페이지·스캔 문항 여부를 확인해 주세요.")
    warnings=file_warnings(exam_bytes,answer_bytes,exam_name,answer_name)
    correct = pd.DataFrame({q: responses[q].eq(key[q]["answer"]) for q in key})
    scores = sum(correct[q].astype(int) * key[q]["points"] for q in key)
    student_rows = []
    for i, name in enumerate(names):
        student_rows.append({"student": name, "score": int(scores[i]),
                             "wrong": [int(q) for q in key if not correct.loc[i, q]]})
    items = [{"number": q, **key[q], "wrong_count": int((~correct[q]).sum()),
              "wrong_rate": float((~correct[q]).mean())} for q in key]
    items.sort(key=lambda x: (-x["wrong_count"], x["number"]))
    minimum = int(scores.min())
    summary = {"student_count": len(names), "question_count": len(key),
               "full_score": sum(item["points"] for item in key.values()),
               "mean": float(scores.mean()), "median": float(scores.median()),
               "std": float(scores.std(ddof=0)), "min": minimum, "max": int(scores.max()),
               "lowest_students": [r["student"] for r in student_rows if r["score"] == minimum],
               "top3_wrong": items[:3]}
    return {"summary": summary, "students": student_rows, "items": items,
            "questions": questions, "key": key, "responses": responses,
            "correct": correct, "warnings": warnings,
            "file_check":{"문제지":len(questions),"정답표":len(key),"OMR":len(responses.columns)}}


def chat_context(result):
    # 평가 문항에 대한 고정 답변이 아니라, 이번 업로드로 계산한 수치입니다.
    worst = result["items"][0]
    return json.dumps({"summary": result["summary"], "students": result["students"],
                       "items": result["items"], "most_wrong_original": {
                           "number": worst["number"], **{k:v for k,v in result["questions"][worst["number"]].items()
                                                          if k != "image_url"}}}, ensure_ascii=False)


def export_excel(result):
    summary = result["summary"]
    summary_rows = [["학생 수", summary["student_count"]], ["문항 수", summary["question_count"]],
                    ["만점", summary["full_score"]], ["평균", summary["mean"]],
                    ["중앙값", summary["median"]], ["표준편차(전체 학생, ddof=0)", summary["std"]],
                    ["최저점", summary["min"]], ["최고점", summary["max"]],
                    ["최저점 학생", ", ".join(summary["lowest_students"])]]
    summary_rows += [[f"오답 상위 {i} 문항", f"{v['number']}번 ({v['wrong_count']}명)"]
                     for i, v in enumerate(summary["top3_wrong"], 1)]
    summary_rows += [["파일 검사 주의",w] for w in result.get('warnings',[])]
    scores = pd.DataFrame({"학생명": [r["student"] for r in result["students"]],
                           "점수": [r["score"] for r in result["students"]]})
    answers = pd.DataFrame({"학생명": scores["학생명"]})
    for q in result["key"]:
        scores[f"Q{q:02d}"] = result["correct"][q].map({True: "O", False: "X"})
        answers[f"Q{q:02d}"] = result["responses"][q].replace(0, "미응답")
    items = pd.DataFrame(result["items"]).rename(columns={"number": "문항", "answer": "정답",
                       "points": "배점", "wrong_count": "오답 학생 수", "wrong_rate": "오답률"})
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame(summary_rows, columns=["항목", "값"]).to_excel(writer, sheet_name="전체 성적 개요", index=False)
        scores.to_excel(writer, sheet_name="학생별 성적", index=False)
        items.to_excel(writer, sheet_name="문항별 분석", index=False)
        answers.to_excel(writer, sheet_name="학생 응답", index=False)
        for sheet in writer.book:
            sheet.freeze_panes = "C2" if sheet.title == "학생별 성적" else "B2"
            sheet.auto_filter.ref = sheet.dimensions
            for cell in sheet[1]:
                cell.fill = PatternFill("solid", fgColor="183C4B")
                cell.font = Font(color="FFFFFF", bold=True)
            for row in sheet.iter_rows(min_row=2):
                for cell in row:
                    # 업로드된 학생명이 Excel 수식으로 실행되지 않게 텍스트로 저장합니다.
                    if cell.data_type == "f":
                        cell.data_type = "s"
                    cell.alignment = Alignment(vertical="center")
                if row[0].row % 2 == 0:
                    for cell in row:
                        cell.fill = PatternFill("solid", fgColor="F1F6F7")
            for i, column in enumerate(sheet.columns, 1):
                width = min(55, max(9, max(len(str(c.value or "")) for c in column) * 1.6 + 2))
                sheet.column_dimensions[get_column_letter(i)].width = width
        for cell in writer.book["문항별 분석"]["E"][1:]:
            cell.number_format = "0.0%"
        for row in writer.book["전체 성적 개요"].iter_rows(min_row=2):
            if row[0].value == "파일 검사 주의":
                row[1].alignment = Alignment(wrap_text=True, vertical="top")
                writer.book["전체 성적 개요"].row_dimensions[row[0].row].height = max(45,15*((len(str(row[1].value))+29)//30))
        for row in [5, 6, 7]:
            writer.book["전체 성적 개요"].cell(row, 2).number_format = "0.00"
    return output.getvalue()
