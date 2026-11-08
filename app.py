import asyncio
import hashlib
import os
import time

import pandas as pd
import streamlit as st

from grading import analyze, export_excel
from assistant import make_agent, answer_chat

st.set_page_config(page_title="클래스 인사이트 | 영어 평가 도우미", page_icon="📘", layout="wide")
st.markdown("""<style>
.stApp {background:#f7f9fa;}
.block-container {max-width:1180px;padding-top:2rem;padding-bottom:3rem;}
h1,h2,h3 {color:#183c4b;letter-spacing:-.04em;}
[data-testid="stSidebar"] {background:#eef3f5;}
[data-testid="stMetric"] {background:white;border:1px solid #e2e9ed;border-radius:14px;padding:18px;}
.brand {font-size:13px;font-weight:800;letter-spacing:.14em;color:#147b78;margin:0 0 26px;}
.hero {background:#183c4b;color:white;border-radius:22px;padding:48px 42px;margin:20px 0 26px;}
.hero h1 {color:white;font-size:44px;line-height:1.25;margin-bottom:20px;}
.hero p {font-size:17px;color:#cadddf;max-width:620px;line-height:1.8;}
.step {background:white;border:1px solid #e2e9ed;border-radius:16px;padding:24px;min-height:180px;}
.step small {color:#14827e;font-weight:800;}
.step h3 {font-size:22px;margin:10px 0;}
.step p {color:#637780;line-height:1.7;font-size:15px;}
</style>""", unsafe_allow_html=True)


def landing():
    st.markdown('<div class="brand">CLASS INSIGHT / 영어 평가 도우미</div>', unsafe_allow_html=True)
    st.markdown('''<div class="hero"><h1>채점 다음의 수업까지,<br>한곳에서 준비하세요.</h1>
    <p>시험지와 정답표, 학생 OMR을 올리면 성적을 정리합니다.<br>
    우리 반의 약점을 살펴보고, 다음 수업에 쓸 새 문제를 만들어 보세요.</p></div>''', unsafe_allow_html=True)
    if st.button("성적 분석 시작하기 →", type="primary"):
        st.switch_page(analysis_page)
    st.write("")
    for col, number, title, detail in zip(st.columns(3), ["01", "02", "03"],
        ["파일 세 개 업로드", "자동 채점과 엑셀", "질문에서 보충 학습까지"],
        ["문제 PDF, 정답 PDF, OMR 엑셀을 준비해 주세요.",
         "학생별 성적과 반 전체 통계, 문항별 정오표를 한 번에 확인합니다.",
         "학생 점수를 물어보고, 많이 틀린 문항을 바탕으로 유사 문제를 만듭니다."]):
        with col:
            st.markdown(f'<div class="step"><small>{number}</small><h3>{title}</h3><p>{detail}</p></div>', unsafe_allow_html=True)
    st.write("")
    st.caption("학생 수와 점수는 업로드한 자료에서 계산합니다. 대화는 현재 세션 안에서 이어집니다.")


def analysis():
    st.markdown('<div class="brand">CLASS INSIGHT / 분석실</div>', unsafe_allow_html=True)
    st.title("우리 반, 이번 시험은 어땠나요?")
    st.caption("왼쪽에서 파일을 업로드하고 분석을 시작해 주세요.")
    with st.sidebar:
        st.subheader("시험 자료 업로드")
        exam = st.file_uploader("문제지 PDF", type=["pdf"], key="exam_upload")
        key = st.file_uploader("정답·배점 PDF", type=["pdf"], key="key_upload")
        omr = st.file_uploader("학생 OMR", type=["xlsx", "csv"], key="omr_upload")
        try:
            saved_key = st.secrets.get("OPENAI_API_KEY", "")
        except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
            saved_key = ""
        api_key = os.environ.get("OPENAI_API_KEY") or saved_key
        if not api_key:
            api_key = st.text_input("OpenAI API 키", type="password", key="api_key_input", help="채팅에 사용합니다. 파일이나 저장소에는 저장하지 않습니다.")
        uploads = [exam, key, omr]
        fingerprint = hashlib.sha256(b"".join(f.getvalue() for f in uploads)).hexdigest() if all(uploads) else None
        # 자료를 바꾸면 이전 학생의 결과와 대화를 즉시 비웁니다.
        if "result" in st.session_state and fingerprint != st.session_state.get("fingerprint"):
            for name in ["result", "excel", "agent", "history", "fingerprint", "agent_key_hash"]:
                st.session_state.pop(name, None)
        start = st.button("분석 시작", type="primary", disabled=not all(uploads), width="stretch")
        if start:
            for name in ["result", "excel", "agent", "history", "fingerprint", "agent_key_hash"]:
                st.session_state.pop(name, None)
            try:
                before = time.perf_counter()
                with st.spinner("정답표를 읽고 채점하고 있습니다…"):
                    result = analyze(exam.getvalue(), key.getvalue(), omr.getvalue(), omr.name)
                    excel = export_excel(result)
                st.session_state.update(result=result, excel=excel, fingerprint=fingerprint, history=[])
                st.success(f"채점 완료 · {time.perf_counter() - before:.2f}초")
            except Exception as exc:
                st.error(str(exc) if isinstance(exc, ValueError) else "파일을 읽지 못했습니다. PDF·OMR 형식을 확인해 주세요.")
        st.download_button("성적 엑셀 다운로드", data=st.session_state.get("excel", b""),
                           file_name="성적_분석.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           disabled="result" not in st.session_state, width="stretch")
        st.caption("OMR: 학생명, Q01, Q02… 열을 사용합니다. 빈 응답/0은 오답으로 채점합니다.")
    result = st.session_state.get("result")
    if result is None:
        st.info("문제지 · 정답표 · 학생 OMR을 올리면 분석 결과와 채팅이 활성화됩니다.")
        st.chat_input("먼저 파일을 올리고 분석을 시작해 주세요.", disabled=True)
        return
    summary = result["summary"]
    for col, title, value in zip(st.columns(4), ["학생 수", "평균", "중앙값", "표준편차"],
                                [f"{summary['student_count']}명", f"{summary['mean']:.2f}점",
                                 f"{summary['median']:.2f}점", f"{summary['std']:.2f}점"]):
        col.metric(title, value)
    st.caption(f"{summary['question_count']}문항 · {summary['full_score']}점 만점 · 표준편차는 업로드된 전체 학생 기준")
    with st.expander("학생별 성적과 많이 틀린 문항 보기"):
        left, right = st.columns([1, 1])
        left.dataframe(pd.DataFrame([{"학생": x["student"], "점수": x["score"], "오답 수": len(x["wrong"])}
                                     for x in result["students"]]), hide_index=True, width="stretch")
        right.dataframe(pd.DataFrame([{"문항": x["number"], "오답 인원": x["wrong_count"], "오답률": f"{x['wrong_rate']:.1%}"}
                                      for x in result["items"][:10]]), hide_index=True, width="stretch")
        st.caption("오답 인원이 같으면 문항 번호 순으로 표시합니다. 듣기 음원이 없는 경우 원래 대본은 확인할 수 없습니다.")
    st.subheader("평가 도우미에게 물어보세요")
    st.caption("예: 평균과 중앙값 알려줘 / 가장 많이 틀린 유형으로 새 문제 만들어줘")
    st.caption("채팅 시 학생 식별자·성적과 필요한 시험 문항이 OpenAI API로 전송됩니다. 업로드 파일은 서버에 따로 저장하지 않습니다.")
    history = st.session_state.setdefault("history", [])
    for message in history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    question = st.chat_input("성적이나 보충 학습에 관해 질문해 주세요.", disabled=not api_key)
    if not api_key:
        st.info("왼쪽에 API 키를 입력하면 채팅을 사용할 수 있습니다. 엑셀 다운로드는 바로 가능합니다.")
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                key_hash = hashlib.sha256(api_key.encode()).hexdigest()
                if st.session_state.get("agent_key_hash") != key_hash:
                    st.session_state.agent = make_agent(result, api_key)
                    st.session_state.agent_key_hash = key_hash
                with st.spinner("답변을 준비하고 있습니다…"):
                    answer, elapsed = asyncio.run(answer_chat(st.session_state.agent, result, history, question))
                st.markdown(answer)
                st.caption(f"응답 시간 {elapsed:.2f}초")
                history.extend([{"role": "user", "content": question}, {"role": "assistant", "content": answer}])
            except (TimeoutError, asyncio.TimeoutError):
                st.warning("응답이 지연되고 있습니다. 잠시 후 다시 질문해 주세요.")
            except Exception:
                st.error("답변을 완료하지 못했습니다. API 키·사용 한도를 확인하거나 다시 질문해 주세요.")


home_page = st.Page(landing, title="서비스 소개", icon="🏠", default=True)
analysis_page = st.Page(analysis, title="성적 분석", icon="📊", url_path="analysis")
st.navigation([home_page, analysis_page]).run()
