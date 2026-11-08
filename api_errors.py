"""API 오류를 원인별로 안내합니다. 원문 오류에는 비밀정보가 있을 수 있어 표시하지 않습니다."""
def api_error_message(exc):
    body = getattr(exc, 'body', None)
    detail = body.get('error', body) if isinstance(body, dict) else {}
    if not isinstance(detail, dict):
        detail = {}
    code = getattr(exc, 'code', None) or detail.get('code')
    kind = detail.get('type')
    status = getattr(exc, 'status_code', None)
    name = type(exc).__name__
    if code in {'insufficient_quota', 'credit_balance_exhausted', 'billing_hard_limit_reached', 'usage_limit_reached', 'billing_not_active'} or kind == 'insufficient_quota':
        return 'OpenAI API의 잔액·결제 또는 사용 한도에 도달했습니다. 자동 재시도는 하지 않습니다. OpenAI 결제·사용 한도를 확인해 주세요. 성적·통계 조회와 엑셀 다운로드는 계속 사용할 수 있습니다.'
    if status == 429 or name == 'RateLimitError':
        return 'OpenAI API의 일시적인 요청 속도 제한입니다. 잠시 후 다시 질문해 주세요. 결제 잔액 부족으로 확인된 오류는 아닙니다.'
    if status == 401 or name == 'AuthenticationError':
        return 'API 키 인증에 실패했습니다. 입력한 키가 유효한지 확인해 주세요.'
    if status == 403:
        return '현재 API 키의 프로젝트 또는 모델 접근 권한을 확인해 주세요.'
    if name in {'APIConnectionError', 'APITimeoutError'}:
        return 'OpenAI 서버 연결이 지연되거나 끊겼습니다. 네트워크 상태를 확인하고 잠시 후 다시 질문해 주세요.'
    if status and status >= 500:
        return 'OpenAI 서버에서 일시적인 오류가 발생했습니다. 잠시 후 다시 질문해 주세요.'
    if status == 400:
        return 'API 요청 형식이나 모델 설정 오류입니다. 사용 한도 문제로 확인된 것은 아닙니다.'
    return '답변 처리 중 오류가 발생했습니다. API 사용 한도 문제로 확인된 것은 아닙니다. 질문과 오류 발생 상황을 확인해 주세요.'
