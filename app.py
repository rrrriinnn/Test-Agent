from api_errors import api_error_message
import asyncio
import base64
from pathlib import Path
import hashlib
import os
import time

import pandas as pd
import streamlit as st

from grading import analyze, export_excel, ENGINE_VERSION
from assistant import make_agent, answer_chat

st.set_page_config(page_title="클래스 인사이트 | 영어 평가 도우미", page_icon="📘", layout="wide")
st.markdown("""<style>
.stApp {background:#fafafa;}
.block-container {max-width:none;padding:4.5rem 1.5rem 2rem;}
.st-key-page_navigation {max-width:340px;margin:0;}
.st-key-page_navigation button {min-height:42px;height:auto;white-space:normal;}
.st-key-page_navigation button p {line-height:1.4;margin:0;overflow:visible;}
@media(max-width:640px) {.block-container {padding:4.5rem 1rem 2rem;}}
h1,h2,h3 {color:#243342;letter-spacing:-.03em;}
[data-testid="stSidebar"] {background:#f1f3f5;border-right:1px solid #d5dbe1;}
[data-testid="stMetric"] {background:white;border:1px solid #e2e7ec;border-radius:8px;padding:12px;}
.welcome {min-height:55vh;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;gap:22px;}
.welcome img {width:170px;max-width:45vw;height:auto;}
.welcome h1 {font-size:30px;font-weight:600;margin:0;}
.welcome p {font-size:15px;color:#647383;line-height:1.8;margin:0;}
[data-testid="stChatMessage"] {background:#f4f6f8;border-radius:10px;}
@media(max-width:640px) {.welcome {min-height:48vh;} .welcome h1 {font-size:25px;}}
</style>""", unsafe_allow_html=True)


def landing():
    logo = Path(__file__).parent / "assets" / "gachon_logo.png"
    logo_data = base64.b64encode(logo.read_bytes()).decode("ascii")
    st.markdown(f'''<div class="welcome">
        <img src="data:image/png;base64,{logo_data}" alt="가천대학교 로고">
        <h1>학원 평가 도우미</h1>
        <p>시험 자료를 업로드하고 성적 분석과 보충 학습을 시작하세요.</p>
        </div>''', unsafe_allow_html=True)


def analysis():
    st.title("학원 평가 도우미")
    with st.sidebar:
        st.subheader("데이터 업로드")
        def clear_uploads():
            for name in ["saved_uploads", "exam_upload", "key_upload", "omr_upload", "result", "excel", "agent", "history", "fingerprint", "agent_key_hash"]:
                st.session_state.pop(name, None)
        exam = st.file_uploader("문제지 PDF", type=["pdf"], key="exam_upload")
        key = st.file_uploader("정답·배점 PDF", type=["pdf"], key="key_upload")
        omr = st.file_uploader("학생 OMR", type=["xlsx", "csv"], key="omr_upload")
        try:
            saved_key = st.secrets.get("OPENAI_API_KEY", "")
        except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
            saved_key = ""
        api_key = os.environ.get("OPENAI_API_KEY") or saved_key
        if not api_key:
            api_key = st.text_input("OpenAI API 키", type="password", value=st.session_state.get("session_api_key", ""), key="api_key_input", help="채팅에 사용합니다. 파일이나 저장소에는 저장하지 않습니다.")
            st.session_state.session_api_key = api_key
        uploads = st.session_state.setdefault("saved_uploads", {})
        for role, file in [("exam", exam), ("key", key), ("omr", omr)]:
            if file is not None:
                uploads[role] = {"name":file.name, "data":file.getvalue()}
        ready = all(role in uploads for role in ["exam", "key", "omr"])
        fingerprint = hashlib.sha256((ENGINE_VERSION + repr([(uploads[role]["name"], hashlib.sha256(uploads[role]["data"]).hexdigest())
            for role in ["exam", "key", "omr"]])).encode()).hexdigest() if ready else None
        if uploads:
            st.caption("세션에 보관 중: " + " · ".join(f["name"] for f in uploads.values()))
            st.caption("페이지를 바꿔도 자료를 유지합니다. 교체하려면 새 파일을 올리고, 삭제하려면 초기화를 누르세요.")
        # 자료를 바꾸면 이전 학생의 결과와 대화를 즉시 비웁니다.
        if "result" in st.session_state and fingerprint != st.session_state.get("fingerprint"):
            for name in ["result", "excel", "agent", "history", "fingerprint", "agent_key_hash"]:
                st.session_state.pop(name, None)
        start = st.button("분석 시작", type="primary", disabled=not ready, width="stretch")
        if start:
            for name in ["result", "excel", "agent", "history", "fingerprint", "agent_key_hash"]:
                st.session_state.pop(name, None)
            try:
                before = time.perf_counter()
                with st.spinner("정답표를 읽고 채점하고 있습니다…"):
                    result = analyze(uploads["exam"]["data"], uploads["key"]["data"], uploads["omr"]["data"], uploads["omr"]["name"],
                                     uploads["exam"]["name"], uploads["key"]["name"])
                    excel = export_excel(result)
                st.session_state.update(result=result, excel=excel, fingerprint=fingerprint, history=[])
                st.success(f"채점 완료 · {time.perf_counter() - before:.2f}초")
            except Exception as exc:
                st.error(str(exc) if isinstance(exc, ValueError) else "파일을 읽지 못했습니다. PDF·OMR 형식을 확인해 주세요.")
        st.download_button("엑셀 추출", data=st.session_state.get("excel", b""),
                           file_name="성적_분석.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           disabled="result" not in st.session_state, width="stretch")
        st.button("자료·대화 초기화", on_click=clear_uploads, width="stretch")
        st.caption("OMR: 학생명, Q01, Q02… 열을 사용합니다. 빈 응답/0은 오답으로 채점합니다.")
    result = st.session_state.get("result")
    if result is None:
        st.subheader("채팅 내역")
        with st.container(height=480, border=True):
            st.info("왼쪽에서 문제지 · 정답표 · 학생 OMR을 올리고 분석 시작을 눌러 주세요.")
        st.chat_input("먼저 파일을 올리고 분석을 시작해 주세요.", disabled=True)
        return
    for warning in result.get("warnings", []):
        st.warning(warning)
    with st.expander("분석 결과 · 학생별 성적 확인", expanded=False):
        summary = result["summary"]
        if result.get("file_check"):
            st.caption("문항 번호 검사 통과 · " + " · ".join(f"{name} {count}문항" for name,count in result["file_check"].items()))
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
    st.subheader("채팅 내역")
    st.caption("예: 평균과 중앙값 알려줘 / 가장 많이 틀린 유형으로 새 문제 만들어줘")
    st.caption("채팅 시 학생 식별자·성적과 필요한 시험 문항이 OpenAI API로 전송됩니다. 업로드 파일은 서버에 따로 저장하지 않습니다.")
    history = st.session_state.setdefault("history", [])
    conversation = st.container(height=480, border=True)
    with conversation:
        if not history:
            st.caption("분석이 완료되었습니다. 학생 점수, 정오 판정, 통계나 보충 문제를 물어보세요.")
        for message in history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    question = st.chat_input("성적이나 보충 학습에 관해 질문해 주세요.", disabled=not api_key)
    if not api_key:
        st.info("왼쪽에 API 키를 입력하면 채팅을 사용할 수 있습니다. 엑셀 다운로드는 바로 가능합니다.")
    if question:
        with conversation:
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
                except Exception as exc:
                    st.error(api_error_message(exc))


home_page = st.Page(landing, title="서비스 소개", icon="🏠", default=True)
analysis_page = st.Page(analysis, title="챗봇", icon="📊", url_path="analysis")
page = st.navigation([home_page, analysis_page], position="hidden")
# 페이지 이동은 항상 본문 맨 위의 가로 버튼을 사용합니다.
with st.container(key="page_navigation"):
    intro_button, chat_button = st.columns(2, gap="small")
    with intro_button:
        if st.button("서비스 소개", key="nav_intro", width="stretch",
                     type="primary" if page.title == "서비스 소개" else "secondary"):
            st.switch_page(home_page)
    with chat_button:
        if st.button("챗봇", key="nav_chat", width="stretch",
                     type="primary" if page.title == "챗봇" else "secondary"):
            st.switch_page(analysis_page)
st.divider()
page.run()
