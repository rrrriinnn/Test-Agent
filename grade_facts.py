"""성적 사실은 모델의 추측 없이 업로드한 응답과 정답을 직접 비교합니다."""
import re
import statistics


def canonical_student(value):
    text = re.sub(r"\s+", "", str(value))
    prefix = re.fullmatch(r"(?:학생|student)#?(\d+)",text,re.I)
    if prefix:
        return prefix.group(1)
    text = re.sub(r"(?:번)?학생$", "", text)
    return re.sub(r"(?<=\d)번$", "", text)


def find_students(result, text):
    explicit = re.findall(r"(?<!\d)(\d+)\s*(?:번\s*)?학생|학번\s*[:：]?\s*(\d+)|student\s*#?\s*(\d+)", text,re.I)
    ids = [next(v for v in group if v) for group in explicit]
    for row in result["students"]:
        name = canonical_student(row["student"])
        if name.isdigit():
            bare_id = re.search(rf"(?<!\d){re.escape(name)}\s*번(?!\s*(?:문제|문항))",text) or re.match(rf"^\s*{re.escape(name)}(?:\s|[,：:])",text)
            if int(name) > max(result["key"]) and bare_id:
                ids.append(name)
        elif re.search(rf"(?<![가-힣A-Za-z0-9]){re.escape(name)}(?:학생|님|은|는|가|이|의|도|랑|와|과|[\s,?!。.]|$)", text):
            ids.append(name)
    if not any(not student.isdigit() for student in ids):
        remainder=re.sub(r"\d+\s*(?:번\s*)?학생", " ",text)
        ids.extend(re.findall(r"학생\s*(\d+)",remainder))
    return list(dict.fromkeys(ids))


def question_numbers(text):
    # 학생 번호와 선택지 번호를 문항 번호로 읽지 않습니다.
    text = re.sub(r"\d+\s*(?:번\s*)?학생|학생\s*\d+|학번\s*[:：]?\s*\d+|student\s*#?\s*\d+", " ", text,flags=re.I)
    text = re.sub(r"(\d+\s*번\s*(?:문제|문항)?\s*(?:에|에서|은|는|의)?\s*)[1-5]\s*번\s*(?:답|보기|선지)?\s*(?:을\s*)?(?:골랐|고른|선택|찍었|체크|응답)", r"\1 ", text)
    numbers = []
    for a,b in re.findall(r"(\d+)\s*[~～∼-]\s*(\d+)\s*번", text):
        if 0 <= int(a) <= int(b) <= 1000:
            numbers.extend(range(int(a),int(b)+1))
    for group in re.findall(r"((?:\d+\s*[,，]\s*)+\d+)\s*번", text):
        numbers.extend(map(int,re.findall(r"\d+",group)))
    for pair in re.findall(r"(?<!\d)(\d+)\s*번|(?:문항|문제|[Qq])\s*(\d+)", text):
        numbers.append(int(next(v for v in pair if v)))
    numbers.extend(map(int,re.findall(r"(?:question|item)\s*#?\s*(\d+)",text,re.I)))
    return list(dict.fromkeys(numbers))


def student_question_fact(result, student, number):
    matches = [(i,r) for i,r in enumerate(result["students"]) if canonical_student(r["student"]) == canonical_student(student)]
    if len(matches) != 1:
        return {"error": "해당 학생이 없거나 이름이 모호합니다. 업로드된 정확한 식별자를 확인해 주세요."}
    if number not in result["key"]:
        return {"error": f"제공된 자료에 {number}번 문항이 없습니다."}
    i,row = matches[0]
    response = int(result["responses"].iloc[i][number])
    answer = result["key"][number]["answer"]
    return {"student":row["student"], "question":number, "response":response,
            "answer":answer, "correct":response==answer, "points":result["key"][number]["points"]}


def render_question_fact(result, student, number):
    fact = student_question_fact(result,student,number)
    if "error" in fact:
        return fact["error"]
    mark = lambda n: "미응답" if n == 0 else f"{'①②③④⑤'[n-1]}({n}번)"
    verdict = "맞았습니다" if fact["correct"] else "틀렸습니다"
    return (f"{fact['student']}의 {number}번은 **{verdict}**. "
            f"학생 응답: {mark(fact['response'])}, 정답: {mark(fact['answer'])}. "
            f"획득 점수: {fact['points'] if fact['correct'] else 0}점.")


def grade_facts_reply(result, history, text):
    """인식한 성적 질문은 직접 답하고, 모호한 정오 질문은 확인을 요청합니다."""
    if re.search(r"만들|생성|출제|해설|지문|번역|대본|스크립트",text):
        return None
    grade_intent = bool(re.search(r"맞|틀|오답|정오|정답|점수|몇\s*점|성적|골랐|고른|선택|응답|찍었|체크|평균|중앙|표준편차|최저|최고|만점|저조|몇\s*개|correct|incorrect|right|wrong|score|chose|answer|points",text,re.I))
    ids = find_students(result,text)
    question_text=text
    for name in ids:
        if not name.isdigit():
            question_text=re.sub(re.escape(name)+r"\s*(?:학생|님)?", " ",question_text)
    numbers = question_numbers(question_text)
    if not grade_intent and not (ids and numbers) and not (numbers and re.search(r"그럼|그 학생|그학생",text)):
        return None
    if re.search(r"출석|출결|참여|어제|지난\s*시험|작년|남학생|여학생",text):
        return None
    roster_question=bool(numbers and re.search(r"학생들|누구|몇\s*명|어떤\s*학생|사람|정답률|오답률",text))
    if not ids and not roster_question and (re.search(r"그\s*학생|그럼|걔|이\s*학생",text) or (numbers and re.search(r"맞|틀|골랐|선택|응답|찍었",text))):
        for previous in reversed(history):
            if previous["role"] == "user":
                ids = find_students(result,previous["content"])
                if ids:
                    break
    selected = []
    for student in ids:
        matches = [(i,r) for i,r in enumerate(result["students"]) if canonical_student(r["student"]) == canonical_student(student)]
        if len(matches) != 1:
            return f"제공된 자료에 {student}학생이 없거나 식별자가 모호합니다. 정확한 학생 이름/번호를 확인해 주세요."
        selected.append(matches[0])
    if selected:
        if numbers:
            return "\n\n".join(render_question_fact(result,r["student"],q) for _,r in selected for q in numbers)
        scores = [r["score"] for _,r in selected]
        if len(scores)==2 and re.search(r"차이|차\s*나",text):
            return f"{selected[0][1]['student']}은 {scores[0]}점, {selected[1][1]['student']}은 {scores[1]}점으로, 차이는 {abs(scores[0]-scores[1])}점입니다."
        if re.search(r"평균|중앙",text):
            return f"지정한 학생들의 평균은 {statistics.mean(scores):.2f}점, 중앙값은 {statistics.median(scores):.2f}점입니다."
        if re.search(r"맞|틀|오답|정답|몇\s*개",text):
            want_wrong = bool(re.search(r"틀|오답",text))
            lines=[]
            for i,row in selected:
                qs=[q for q,k in result['key'].items() if (int(result['responses'].iloc[i][q]) == k['answer']) != want_wrong]
                lines.append(f"{row['student']}의 {'틀린' if want_wrong else '맞은'} 문항은 {len(qs)}개입니다. " +
                             (", ".join(f"{q}번" for q in qs) if qs else "해당 문항이 없습니다."))
            return "\n\n".join(lines)
        return "\n".join(f"{r['student']}은 **{r['score']}점**입니다." for _,r in selected)
    if numbers and re.search(r"학생|사람|몇\s*명|누구|정답률|오답률",text):
        wrong = bool(re.search(r"틀|오답",text))
        lines=[]
        for q in numbers:
            if q not in result['key']:
                lines.append(f"제공된 자료에 {q}번 문항이 없습니다.");continue
            names=[r['student'] for i,r in enumerate(result['students']) if (int(result['responses'].iloc[i][q]) == result['key'][q]['answer']) != wrong]
            rate=f" ({len(names)/len(result['students']):.1%})" if re.search(r"정답률|오답률",text) else ""
            lines.append(f"{q}번을 {'틀린' if wrong else '맞은'} 학생은 {len(names)}명{rate}입니다. " + (", ".join(names) if names else "해당 학생이 없습니다."))
        return "\n\n".join(lines)
    if numbers and re.search(r"맞|틀|골랐|선택|응답|찍었",text):
        return "어느 학생의 응답인지 확인할 수 없습니다. 학생 이름/번호와 문항 번호를 함께 알려주세요."
    if numbers and '정답' in text:
        return "\n".join(f"{q}번의 정답은 {'①②③④⑤'[result['key'][q]['answer']-1]}입니다." if q in result['key'] else f"제공된 자료에 {q}번 문항이 없습니다." for q in numbers)
    summary=result['summary']
    # 표준편차는 흩어짐의 크기이며, 학생의 거리는 평균에서의 절대 차이입니다.
    distance_far = re.search(r"멀|먼|벗어|떨어|편차.*(?:큰|최대)", text)
    distance_close = re.search(r"가까|근접|비슷|유사|편차.*(?:작|적|최소)", text)
    if not (distance_far or distance_close) and re.search(r"그\s*학생|누구|몇\s*번", text):
        previous = next((m['content'] for m in reversed(history) if m['role']=='user'), '')
        if re.search(r"표준편차|평균|중앙", previous):
            distance_far = re.search(r"멀|먼|벗어|떨어|편차.*(?:큰|최대)", previous)
            distance_close = re.search(r"가까|근접|비슷|유사|편차.*(?:작|적|최소)", previous)
    if (distance_far or distance_close) and (re.search(r"표준편차|평균|중앙|점수", text) or re.search(r"그\s*학생|누구|몇\s*번", text)):
        scores = [float(row['score']) for row in result['students']]
        mean = statistics.mean(scores)
        median = statistics.median(scores)
        std = statistics.pstdev(scores)
        if distance_far:
            distance = max(abs(score - mean) for score in scores)
            farthest = [row for row in result['students'] if abs(abs(float(row['score']) - mean) - distance) < 1e-9]
            people = ', '.join(f"{row['student']}({row['score']}점)" for row in farthest)
            return (f"평균에서 점수가 가장 멀리 떨어진 학생은 **{people}**입니다. "
                    f"전체 평균은 {mean:.2f}점이고, 평균과의 차이는 **{distance:.2f}점**입니다. "
                    f"전체 학생 기준 표준편차는 {std:.2f}점입니다.")
        if re.search(r"표준편차", text) or (not re.search(r"평균|중앙", text) and re.search(r"표준편차", previous if 'previous' in locals() else '')):
            diff_from_mean_std = min(abs(abs(score - mean) - std) for score in scores)
            closest_std_dev = [row for row in result['students'] if abs(abs( abs(float(row['score']) - mean) - std ) - diff_from_mean_std) < 1e-9]
            people_std = ', '.join(f"{row['student']}({row['score']}점, 평균과의 편차 {abs(float(row['score']) - mean):.2f}점)" for row in closest_std_dev)
            diff_from_mean = min(abs(score - mean) for score in scores)
            closest_mean = [row for row in result['students'] if abs(abs(float(row['score']) - mean) - diff_from_mean) < 1e-9]
            people_mean = ', '.join(f"{row['student']}({row['score']}점)" for row in closest_mean)
            return (f"전체 학생 기준 표준편차는 **{std:.2f}점**(전체 평균 {mean:.2f}점)입니다.\n"
                    f"- 평균과의 점수 차이(편차)가 표준편차({std:.2f}점)에 가장 가까운 학생: **{people_std}**\n"
                    f"- 평균({mean:.2f}점) 자체에 가장 가까운(편차가 가장 작은) 학생: **{people_mean}** (차이 {diff_from_mean:.2f}점)")
        target_val = median if re.search(r"중앙", text) else mean
        target_name = "중앙값" if re.search(r"중앙", text) else "평균"
        distance = min(abs(score - target_val) for score in scores)
        closest = [row for row in result['students'] if abs(abs(float(row['score']) - target_val) - distance) < 1e-9]
        people = ', '.join(f"{row['student']}({row['score']}점)" for row in closest)
        return (f"{target_name}({target_val:.2f}점)에 점수가 가장 가까운 학생은 **{people}**입니다 "
                f"(차이 **{distance:.2f}점**).")
    if re.search(r"평균|중앙|표준편차", text) and not re.search(r"누구|어떤\s*학생|어느\s*학생|몇\s*번\s*학생|몇\s*명|이름|순위|등수|비교|차이|가까|멀|먼|높|낮|이상|이하|초과|미만", text):
        parts=[]
        for pattern,key,label in [('평균','mean','평균'),('중앙','median','중앙값'),('표준편차','std','표준편차(전체 학생 기준)')]:
            if re.search(pattern,text):parts.append(f"{label} {summary[key]:.2f}점")
        return "이번 시험은 " + ", ".join(parts) + "입니다."
    if re.search(r"최저|저조|가장.*낮",text):
        return f"최저점은 {summary['min']}점이며, 해당 학생은 {', '.join(summary['lowest_students'])}입니다."
    return None
